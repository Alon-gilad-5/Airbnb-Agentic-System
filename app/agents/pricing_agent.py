"""Deterministic pricing recommendations over structured listing and market data."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import logging
import math
import re
from statistics import mean
import time
from typing import Any, Literal

from app.agents.base import Agent, AgentResult
from app.schemas import PricingRecommendation, PricingResponse, PricingSignalSummary, StepLog
from app.services.chat_service import ChatService
from app.services.listing_store import PROPERTY_SPEC_COLUMNS, REVIEW_SCORE_COLUMNS, REVIEW_VOLUME_COLUMNS


PRICE_MODES = {"recommended", "conservative", "aggressive"}
MARKET_SIGNAL_WARN_MS = 5_000
logger = logging.getLogger(__name__)


@dataclass
class PricingAgentConfig:
    default_horizon_days: int = 14
    max_horizon_days: int = 30
    low_conf_cap_pct: float = 5.0
    medium_conf_cap_pct: float = 8.0
    high_conf_cap_pct: float = 10.0
    event_radius_km: int = 15
    strong_event_threshold: int = 2
    weather_soft_threshold_days: int = 2
    storm_wind_kph_threshold: float = 45.0
    heavy_rain_mm_threshold: float = 20.0
    snow_cm_threshold: float = 4.0
    review_volume_adjustment_pct: float = 1.5


@dataclass
class PricingRunOutcome:
    narrative: str
    error: str | None
    recommendation: PricingRecommendation | None
    signals: PricingSignalSummary | None
    steps: list[StepLog]


def _context_str(context: dict[str, object], key: str) -> str | None:
    value = context.get(key)
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return None


def _context_float(context: dict[str, object], key: str) -> float | None:
    value = context.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _context_int(context: dict[str, object], key: str) -> int | None:
    value = context.get(key)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _round_price(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value, 2)


def _classify_confidence(level_index: int) -> Literal["low", "medium", "high"]:
    if level_index <= 0:
        return "low"
    if level_index == 1:
        return "medium"
    return "high"


class PricingAgent(Agent):
    """Recommend one nightly price using deterministic rules and optional LLM explanation."""

    name = "pricing_agent"

    def __init__(
        self,
        *,
        listing_store: Any | None,
        neighbor_store: Any | None,
        market_data_providers: Any,
        chat_service: ChatService,
        config: PricingAgentConfig,
    ) -> None:
        self._listing_store = listing_store
        self._neighbor_store = neighbor_store
        self._providers = market_data_providers
        self._chat = chat_service
        self.config = config

    def run(self, prompt: str, context: dict[str, object] | None = None) -> AgentResult:
        outcome = self.recommend(prompt, context=context)
        return AgentResult(response=outcome.narrative, steps=outcome.steps)

    def recommend(self, prompt: str, *, context: dict[str, object] | None = None) -> PricingRunOutcome:
        ctx = context or {}
        steps: list[StepLog] = []
        resolved_context = self._resolve_context(ctx)
        steps.append(
            StepLog(
                module="pricing_agent.context_resolve",
                prompt={"context_keys": sorted(ctx.keys())},
                response=resolved_context,
            )
        )

        property_id = resolved_context.get("property_id")
        if not property_id:
            return self._error_outcome("Property ID is required for pricing recommendations.", steps)
        if self._neighbor_store is None:
            return self._error_outcome("Neighbor store is unavailable. Check DATABASE_URL and retry.", steps)
        if self._listing_store is None:
            return self._error_outcome("Listing store is unavailable. Check DATABASE_URL and retry.", steps)

        try:
            neighbor_ids = self._neighbor_store.get_neighbors(str(property_id)) or []
        except Exception as exc:
            return self._error_outcome(f"Neighbor lookup failed: {type(exc).__name__}: {exc}", steps)

        neighbor_ids = [neighbor_id for neighbor_id in neighbor_ids if neighbor_id and neighbor_id != property_id]
        steps.append(
            StepLog(
                module="pricing_agent.neighbor_lookup",
                prompt={"property_id": property_id},
                response={"neighbor_count": len(neighbor_ids), "neighbor_ids": neighbor_ids[:10]},
            )
        )
        if not neighbor_ids:
            return self._error_outcome(f"No neighbors found for property_id={property_id}.", steps)

        selected_columns = [*PROPERTY_SPEC_COLUMNS, *REVIEW_SCORE_COLUMNS, *REVIEW_VOLUME_COLUMNS]
        try:
            rows = self._listing_store.get_listings_by_ids([property_id, *neighbor_ids], selected_columns)
        except Exception as exc:
            return self._error_outcome(f"Listing fetch failed: {type(exc).__name__}: {exc}", steps)

        rows_by_id = {str(row.get("id", "")).strip(): row for row in rows if str(row.get("id", "")).strip()}
        owner_row = rows_by_id.get(property_id)
        neighbor_rows = [rows_by_id[neighbor_id] for neighbor_id in neighbor_ids if neighbor_id in rows_by_id]
        if owner_row is None:
            return self._error_outcome(f"Owner property {property_id} is missing from large_dataset_table.", steps)
        if not neighbor_rows:
            return self._error_outcome(
                f"No neighbor listing rows were found in large_dataset_table for property_id={property_id}.",
                steps,
            )

        price_stats = self._compute_price_position(owner_row, neighbor_rows)
        quality_stats = self._compute_quality_position(owner_row, neighbor_rows)
        review_volume_stats = self._compute_review_volume_stats(owner_row, neighbor_rows)
        steps.append(
            StepLog(
                module="pricing_agent.data_fetch",
                prompt={
                    "property_id": property_id,
                    "requested_columns": selected_columns,
                    "requested_listing_count": len(neighbor_ids) + 1,
                },
                response={
                    "rows_returned": len(rows),
                    "owner_found": True,
                    "neighbor_rows_found": len(neighbor_rows),
                    "current_price": price_stats["current_price"],
                    "neighbor_avg_price": price_stats["neighbor_avg_price"],
                    "review_score_gap": quality_stats["review_score_gap"],
                    "owner_number_of_reviews": review_volume_stats["owner_number_of_reviews"],
                    "neighbor_avg_number_of_reviews": review_volume_stats["neighbor_avg_number_of_reviews"],
                    "owner_recent_reviews_30d": review_volume_stats["owner_recent_reviews_30d"],
                    "neighbor_avg_recent_reviews_30d": review_volume_stats["neighbor_avg_recent_reviews_30d"],
                    "neighbor_review_count_coverage": review_volume_stats["neighbor_review_count_coverage"],
                    "missing_review_volume_fields": review_volume_stats["missing_fields"],
                },
            )
        )

        market_stats = self._compute_market_signals(
            latitude=resolved_context.get("latitude"),
            longitude=resolved_context.get("longitude"),
            horizon_days=int(resolved_context["horizon_days"]),
        )
        self._log_market_signal_fetch(
            property_id=str(property_id),
            horizon_days=int(resolved_context["horizon_days"]),
            market_stats=market_stats,
        )
        steps.append(
            StepLog(
                module="pricing_agent.market_signal_fetch",
                prompt={
                    "latitude": resolved_context.get("latitude"),
                    "longitude": resolved_context.get("longitude"),
                    "horizon_days": resolved_context["horizon_days"],
                },
                response=market_stats,
            )
        )

        confidence = self._compute_confidence(
            price_stats=price_stats,
            quality_stats=quality_stats,
            review_volume_stats=review_volume_stats,
            market_stats=market_stats,
            neighbor_count=len(neighbor_rows),
        )
        recommendation = self._compute_recommendation(
            price_stats=price_stats,
            quality_stats=quality_stats,
            review_volume_stats=review_volume_stats,
            market_stats=market_stats,
            price_mode=str(resolved_context["price_mode"]),
            confidence=confidence,
        )
        steps.append(
            StepLog(
                module="pricing_agent.recommendation_compute",
                prompt={
                    "price_mode": resolved_context["price_mode"],
                    "confidence": confidence,
                    "market_pressure": market_stats["market_pressure"],
                },
                response={
                    "base_price_action": recommendation["base_price_action"],
                    "base_price_change_pct": recommendation["base_price_change_pct"],
                    "review_volume_position": review_volume_stats["review_volume_strength"],
                    "review_volume_adjustment_pct": recommendation["review_volume_adjustment_pct"],
                    "final_price_change_pct": recommendation["final_price_change_pct"],
                    "price_action": recommendation["price_action"],
                    "recommended_price": recommendation["recommended_price"],
                    "confidence": confidence,
                },
            )
        )

        risk_note = self._build_risk_note(
            price_stats=price_stats,
            market_stats=market_stats,
            review_volume_stats=review_volume_stats,
            confidence=confidence,
        )
        pricing_recommendation = PricingRecommendation(
            current_price=price_stats["current_price"],
            recommended_price=recommendation["recommended_price"],
            price_change_abs=(
                round(recommendation["recommended_price"] - price_stats["current_price"], 2)
                if recommendation["recommended_price"] is not None and price_stats["current_price"] is not None
                else None
            ),
            price_change_pct=recommendation["final_price_change_pct"],
            price_action=recommendation["price_action"],
            confidence=confidence,
            primary_reason=self._build_primary_reason(
                recommendation["price_action"],
                price_stats=price_stats,
                quality_stats=quality_stats,
                review_volume_stats=review_volume_stats,
                market_stats=market_stats,
            ),
            risk_note=risk_note,
        )
        signal_summary = PricingSignalSummary(
            neighbor_avg_price=price_stats["neighbor_avg_price"],
            neighbor_min_price=price_stats["neighbor_min_price"],
            neighbor_max_price=price_stats["neighbor_max_price"],
            price_position_pct=price_stats["price_position_pct"],
            review_score_gap=quality_stats["review_score_gap"],
            strongest_review_metric=quality_stats["strongest_review_metric"],
            weakest_review_metric=quality_stats["weakest_review_metric"],
            demand_signal_count=market_stats["demand_signal_count"],
            high_severity_signal_count=market_stats["high_severity_signal_count"],
            market_pressure=market_stats["market_pressure"],
            owner_number_of_reviews=review_volume_stats["owner_number_of_reviews"],
            neighbor_avg_number_of_reviews=review_volume_stats["neighbor_avg_number_of_reviews"],
            owner_recent_reviews_30d=review_volume_stats["owner_recent_reviews_30d"],
            neighbor_avg_recent_reviews_30d=review_volume_stats["neighbor_avg_recent_reviews_30d"],
            review_volume_strength=review_volume_stats["review_volume_strength"],
        )

        response_text = self._build_narrative(
            prompt=prompt,
            recommendation=pricing_recommendation,
            signals=signal_summary,
        )
        steps.append(self._last_answer_step)
        return PricingRunOutcome(
            narrative=response_text,
            error=None,
            recommendation=pricing_recommendation,
            signals=signal_summary,
            steps=steps,
        )

    @property
    def _last_answer_step(self) -> StepLog:
        return self.__last_answer_step

    @_last_answer_step.setter
    def _last_answer_step(self, step: StepLog) -> None:
        self.__last_answer_step = step

    def _error_outcome(self, message: str, steps: list[StepLog]) -> PricingRunOutcome:
        self._last_answer_step = StepLog(
            module="pricing_agent.answer_generation",
            prompt={"status": "skipped"},
            response={"status": "skipped", "reason": message},
        )
        return PricingRunOutcome(
            narrative=message,
            error=message,
            recommendation=None,
            signals=None,
            steps=steps + [self._last_answer_step],
        )

    def _resolve_context(self, context: dict[str, object]) -> dict[str, object]:
        horizon_days = _context_int(context, "horizon_days") or self.config.default_horizon_days
        horizon_days = max(1, min(horizon_days, self.config.max_horizon_days))
        price_mode = _context_str(context, "price_mode")
        if price_mode not in PRICE_MODES:
            price_mode = "recommended"
        return {
            "property_id": _context_str(context, "property_id"),
            "property_name": _context_str(context, "property_name"),
            "latitude": _context_float(context, "latitude"),
            "longitude": _context_float(context, "longitude"),
            "horizon_days": horizon_days,
            "price_mode": price_mode,
        }

    def _compute_price_position(self, owner_row: dict[str, Any], neighbor_rows: list[dict[str, Any]]) -> dict[str, Any]:
        current_price = self._normalize_numeric("price", owner_row.get("price"))
        neighbor_prices = [
            price
            for price in (self._normalize_numeric("price", row.get("price")) for row in neighbor_rows)
            if price is not None
        ]
        neighbor_avg = mean(neighbor_prices) if neighbor_prices else None
        gap_pct = None
        if current_price is not None and neighbor_avg is not None and not math.isclose(neighbor_avg, 0.0, abs_tol=1e-9):
            gap_pct = ((current_price - neighbor_avg) / neighbor_avg) * 100.0
        price_position_pct = None
        if current_price is not None and neighbor_prices:
            lower_or_equal = len([price for price in neighbor_prices if price <= current_price])
            price_position_pct = round((lower_or_equal / len(neighbor_prices)) * 100.0, 1)
        return {
            "current_price": _round_price(current_price),
            "neighbor_avg_price": _round_price(neighbor_avg),
            "neighbor_min_price": _round_price(min(neighbor_prices)) if neighbor_prices else None,
            "neighbor_max_price": _round_price(max(neighbor_prices)) if neighbor_prices else None,
            "price_gap_pct": round(gap_pct, 2) if gap_pct is not None else None,
            "price_position_pct": price_position_pct,
            "usable_neighbor_prices": len(neighbor_prices),
        }

    def _compute_quality_position(self, owner_row: dict[str, Any], neighbor_rows: list[dict[str, Any]]) -> dict[str, Any]:
        owner_rating = self._normalize_numeric("review_scores_rating", owner_row.get("review_scores_rating"))
        neighbor_rating_values = [
            value
            for value in (
                self._normalize_numeric("review_scores_rating", row.get("review_scores_rating"))
                for row in neighbor_rows
            )
            if value is not None
        ]
        neighbor_avg_rating = mean(neighbor_rating_values) if neighbor_rating_values else None
        review_score_gap = None
        if owner_rating is not None and neighbor_avg_rating is not None:
            review_score_gap = round(owner_rating - neighbor_avg_rating, 3)

        metric_deltas: list[tuple[str, float]] = []
        for column in REVIEW_SCORE_COLUMNS:
            owner_value = self._normalize_numeric(column, owner_row.get(column))
            neighbor_values = [
                value
                for value in (self._normalize_numeric(column, row.get(column)) for row in neighbor_rows)
                if value is not None
            ]
            if owner_value is None or not neighbor_values:
                continue
            metric_deltas.append((column, owner_value - mean(neighbor_values)))

        strongest = max(metric_deltas, key=lambda item: item[1])[0] if metric_deltas else None
        weakest = min(metric_deltas, key=lambda item: item[1])[0] if metric_deltas else None
        return {
            "owner_rating": owner_rating,
            "neighbor_avg_rating": round(neighbor_avg_rating, 3) if neighbor_avg_rating is not None else None,
            "review_score_gap": review_score_gap,
            "strongest_review_metric": strongest,
            "weakest_review_metric": weakest,
        }

    def _compute_review_volume_stats(self, owner_row: dict[str, Any], neighbor_rows: list[dict[str, Any]]) -> dict[str, Any]:
        owner_total = self._normalize_int(owner_row.get("number_of_reviews"))
        owner_recent = self._normalize_int(owner_row.get("number_of_reviews_l30d"))
        owner_ltm = self._normalize_int(owner_row.get("number_of_reviews_ltm"))
        owner_recent_fallback = self._normalize_float(owner_row.get("reviews_per_month"))

        neighbor_totals = [value for value in (self._normalize_int(row.get("number_of_reviews")) for row in neighbor_rows) if value is not None]
        neighbor_recent = [
            value
            for value in (self._normalize_int(row.get("number_of_reviews_l30d")) for row in neighbor_rows)
            if value is not None
        ]
        neighbor_momentum_values = []
        for row in neighbor_rows:
            momentum = self._recent_review_momentum(row)
            if momentum is not None:
                neighbor_momentum_values.append(momentum)

        review_depth_score = math.log1p(owner_total) if owner_total is not None else 0.0
        recent_review_momentum = self._recent_review_momentum(owner_row) or 0.0
        total_ratio = self._ratio(owner_total, mean(neighbor_totals) if neighbor_totals else None)
        recent_ratio = self._ratio(recent_review_momentum, mean(neighbor_momentum_values) if neighbor_momentum_values else None)
        usable_ratios = [ratio for ratio in (total_ratio, recent_ratio) if ratio is not None]
        review_volume_strength: Literal["below_market", "in_line", "above_market"] | None = None
        avg_ratio = mean(usable_ratios) if usable_ratios else None
        if avg_ratio is not None:
            if avg_ratio >= 1.2:
                review_volume_strength = "above_market"
            elif avg_ratio <= 0.8:
                review_volume_strength = "below_market"
            else:
                review_volume_strength = "in_line"

        return {
            "owner_number_of_reviews": owner_total,
            "neighbor_avg_number_of_reviews": round(mean(neighbor_totals), 2) if neighbor_totals else None,
            "owner_recent_reviews_30d": owner_recent,
            "neighbor_avg_recent_reviews_30d": round(mean(neighbor_recent), 2) if neighbor_recent else None,
            "review_depth_score": round(review_depth_score, 3),
            "recent_review_momentum": round(recent_review_momentum, 3),
            "review_volume_strength": review_volume_strength,
            "owner_number_of_reviews_ltm": owner_ltm,
            "neighbor_review_count_coverage": len(neighbor_totals),
            "missing_fields": {
                "owner_total_reviews_missing": int(owner_total is None),
                "owner_recent_reviews_missing": int(owner_recent is None and owner_recent_fallback is None),
                "neighbor_total_reviews_missing": sum(
                    1 for row in neighbor_rows if self._normalize_int(row.get("number_of_reviews")) is None
                ),
                "neighbor_recent_reviews_missing": sum(
                    1
                    for row in neighbor_rows
                    if self._normalize_int(row.get("number_of_reviews_l30d")) is None
                    and self._normalize_float(row.get("reviews_per_month")) is None
                ),
            },
        }

    def _compute_market_signals(
        self,
        *,
        latitude: float | None,
        longitude: float | None,
        horizon_days: int,
    ) -> dict[str, Any]:
        now = datetime.now(tz=UTC)
        end = now + timedelta(days=horizon_days)
        holidays: list[Any] = []
        holiday_meta: list[dict[str, Any]] = []
        holiday_elapsed_ms = 0
        for year in sorted({now.year, end.year}):
            started = time.perf_counter()
            year_holidays, meta = self._providers.fetch_us_public_holidays(year=year)
            holiday_elapsed_ms += int((time.perf_counter() - started) * 1000)
            holiday_meta.append(meta)
            holidays.extend(year_holidays)
        holiday_count = len([holiday for holiday in holidays if now.date() <= holiday.day <= end.date()])

        severe_weather_days = 0
        weather_meta: dict[str, Any] = {"status": "skipped", "reason": "missing_coordinates"}
        events_meta: dict[str, Any] = {"status": "skipped", "reason": "missing_coordinates"}
        events: list[Any] = []
        weather_elapsed_ms = 0
        events_elapsed_ms = 0
        if latitude is not None and longitude is not None:
            started = time.perf_counter()
            weather_days, weather_meta = self._providers.fetch_weather_forecast(
                latitude=latitude,
                longitude=longitude,
                lookahead_days=horizon_days,
            )
            weather_elapsed_ms = int((time.perf_counter() - started) * 1000)
            severe_weather_days = len(
                [
                    day
                    for day in weather_days
                    if (day.wind_kph_max or 0) >= self.config.storm_wind_kph_threshold
                    or (day.precipitation_mm or 0) >= self.config.heavy_rain_mm_threshold
                    or (day.snowfall_cm or 0) >= self.config.snow_cm_threshold
                ]
            )
            started = time.perf_counter()
            events, events_meta = self._providers.fetch_ticketmaster_events(
                latitude=latitude,
                longitude=longitude,
                radius_km=self.config.event_radius_km,
                start_at_utc=now,
                end_at_utc=end,
            )
            events_elapsed_ms = int((time.perf_counter() - started) * 1000)

        strong_event_count = len([event for event in events if (event.popularity_hint or "").lower() == "high"])
        demand_signal_count = len(events) + holiday_count
        high_severity_signal_count = strong_event_count + holiday_count
        if severe_weather_days >= self.config.weather_soft_threshold_days and demand_signal_count <= 1:
            market_pressure: Literal["soft", "neutral", "strong"] = "soft"
        elif strong_event_count >= 1 or len(events) >= self.config.strong_event_threshold or holiday_count >= 1:
            market_pressure = "strong"
        elif severe_weather_days > 0:
            market_pressure = "soft"
        else:
            market_pressure = "neutral"
        return {
            "market_pressure": market_pressure,
            "demand_signal_count": demand_signal_count,
            "high_severity_signal_count": high_severity_signal_count,
            "severe_weather_days": severe_weather_days,
            "event_count": len(events),
            "strong_event_count": strong_event_count,
            "holiday_count": holiday_count,
            "weather_status": weather_meta.get("status"),
            "events_status": events_meta.get("status"),
            "holiday_statuses": [meta.get("status") for meta in holiday_meta],
            "weather_elapsed_ms": weather_elapsed_ms,
            "events_elapsed_ms": events_elapsed_ms,
            "holiday_elapsed_ms": holiday_elapsed_ms,
        }

    def _log_market_signal_fetch(
        self,
        *,
        property_id: str,
        horizon_days: int,
        market_stats: dict[str, Any],
    ) -> None:
        holiday_statuses = [status for status in market_stats.get("holiday_statuses", []) if status]
        component_statuses = {
            "weather": market_stats.get("weather_status"),
            "events": market_stats.get("events_status"),
            "holidays": ",".join(holiday_statuses) if holiday_statuses else None,
        }
        issue_statuses = {
            name: status
            for name, status in component_statuses.items()
            if status not in {None, "ok"}
        }
        timings = {
            "weather": int(market_stats.get("weather_elapsed_ms") or 0),
            "events": int(market_stats.get("events_elapsed_ms") or 0),
            "holidays": int(market_stats.get("holiday_elapsed_ms") or 0),
        }
        slow_components = {
            name: elapsed
            for name, elapsed in timings.items()
            if elapsed >= MARKET_SIGNAL_WARN_MS
        }
        if issue_statuses or slow_components:
            logger.warning(
                "pricing_agent market signal fetch property_id=%s horizon_days=%s statuses=%s timings_ms=%s",
                property_id,
                horizon_days,
                issue_statuses or component_statuses,
                timings,
            )

    def _compute_confidence(
        self,
        *,
        price_stats: dict[str, Any],
        quality_stats: dict[str, Any],
        review_volume_stats: dict[str, Any],
        market_stats: dict[str, Any],
        neighbor_count: int,
    ) -> Literal["low", "medium", "high"]:
        usable_price = price_stats["current_price"] is not None and price_stats["neighbor_avg_price"] is not None
        usable_reviews = quality_stats["review_score_gap"] is not None
        clear_market_signal = market_stats["market_pressure"] != "neutral" or market_stats["demand_signal_count"] > 0
        level = 0
        if usable_price and neighbor_count >= 3:
            level = 1
        if usable_price and usable_reviews and neighbor_count >= 5 and clear_market_signal:
            level = 2

        review_strength = review_volume_stats["review_volume_strength"]
        owner_total_reviews = review_volume_stats["owner_number_of_reviews"]
        has_recent_activity = (review_volume_stats["recent_review_momentum"] or 0.0) > 0
        quality_gap = quality_stats["review_score_gap"] or 0.0
        if (
            owner_total_reviews is not None
            and review_strength == "above_market"
            and has_recent_activity
            and quality_gap >= -0.02
            and neighbor_count >= 5
            and level == 1
        ):
            level += 1
        if (
            review_strength == "below_market"
            or (quality_gap > 0.05 and (owner_total_reviews or 0) < 10)
            or ((review_volume_stats["recent_review_momentum"] or 0.0) <= 0 and review_strength != "above_market")
        ):
            level -= 1
        level = max(0, min(level, 2))
        return _classify_confidence(level)

    def _compute_recommendation(
        self,
        *,
        price_stats: dict[str, Any],
        quality_stats: dict[str, Any],
        review_volume_stats: dict[str, Any],
        market_stats: dict[str, Any],
        price_mode: str,
        confidence: Literal["low", "medium", "high"],
    ) -> dict[str, Any]:
        gap_pct = price_stats["price_gap_pct"]
        current_price = price_stats["current_price"]
        neighbor_avg = price_stats["neighbor_avg_price"]
        quality_gap = quality_stats["review_score_gap"] or 0.0
        market_pressure = market_stats["market_pressure"]
        materially_below = gap_pct is not None and gap_pct <= -3.0
        materially_above = gap_pct is not None and gap_pct >= 3.0
        quality_below = quality_gap <= -0.05

        base_action: Literal["raise", "hold", "lower", "unknown"] = "hold"
        base_pct = 0.0
        if current_price is None or neighbor_avg is None:
            base_action = "unknown"
        elif market_pressure == "strong" and (materially_below or current_price <= neighbor_avg) and not quality_below:
            base_action, base_pct = "raise", 6.0
        elif market_pressure == "neutral" and materially_below and not quality_below:
            base_action, base_pct = "raise", 5.0
        elif materially_above and (quality_below or market_pressure == "soft"):
            base_action, base_pct = "lower", -6.0
        elif materially_above and market_pressure != "strong":
            base_action, base_pct = "lower", -4.0
        elif quality_below and not materially_below:
            base_action, base_pct = "lower", -3.0
        elif market_pressure == "soft" and current_price >= neighbor_avg:
            base_action, base_pct = "lower", -4.0

        adjustment = self._review_volume_adjustment(
            base_action=base_action,
            review_volume_strength=review_volume_stats["review_volume_strength"],
            recent_review_momentum=review_volume_stats["recent_review_momentum"],
            quality_gap=quality_gap,
        )
        final_pct = base_pct + adjustment
        if price_mode == "conservative":
            final_pct *= 0.5
        cap_by_conf = {
            "low": self.config.low_conf_cap_pct,
            "medium": self.config.medium_conf_cap_pct,
            "high": self.config.high_conf_cap_pct,
        }[confidence]
        final_pct = _clamp(final_pct, -cap_by_conf, cap_by_conf)
        if math.isclose(final_pct, 0.0, abs_tol=0.1):
            final_pct = 0.0
            price_action: Literal["raise", "hold", "lower", "unknown"] = "hold" if base_action != "unknown" else "unknown"
        elif final_pct > 0:
            price_action = "raise"
        else:
            price_action = "lower"

        recommended_price = None
        if current_price is not None:
            recommended_price = _round_price(current_price * (1.0 + final_pct / 100.0))
        elif neighbor_avg is not None:
            recommended_price = neighbor_avg
        return {
            "base_price_action": base_action,
            "base_price_change_pct": round(base_pct, 2),
            "review_volume_adjustment_pct": round(adjustment, 2),
            "final_price_change_pct": round(final_pct, 2) if recommended_price is not None or base_action != "unknown" else None,
            "price_action": price_action,
            "recommended_price": recommended_price,
        }

    def _review_volume_adjustment(
        self,
        *,
        base_action: Literal["raise", "hold", "lower", "unknown"],
        review_volume_strength: Literal["below_market", "in_line", "above_market"] | None,
        recent_review_momentum: float,
        quality_gap: float,
    ) -> float:
        max_adjustment = self.config.review_volume_adjustment_pct
        if base_action in {"hold", "unknown"} or review_volume_strength is None:
            return 0.0
        high_recent = recent_review_momentum > 0.5
        if base_action == "raise":
            if review_volume_strength == "above_market":
                return max_adjustment if high_recent else 0.5
            if review_volume_strength == "below_market":
                return -max_adjustment
            return 0.0
        if review_volume_strength == "above_market":
            return max_adjustment
        if review_volume_strength == "below_market" and quality_gap < -0.05:
            return -0.5
        return 0.0

    def _build_primary_reason(
        self,
        price_action: Literal["raise", "hold", "lower", "unknown"],
        *,
        price_stats: dict[str, Any],
        quality_stats: dict[str, Any],
        review_volume_stats: dict[str, Any],
        market_stats: dict[str, Any],
    ) -> str:
        price_gap = price_stats["price_gap_pct"]
        quality_gap = quality_stats["review_score_gap"] or 0.0
        review_strength = review_volume_stats["review_volume_strength"]
        if price_action == "raise":
            parts = ["You are priced below nearby comps."]
            if quality_gap >= 0.03:
                parts.append("Your review position is at or above market.")
            if market_stats["market_pressure"] == "strong":
                parts.append("Short-horizon demand signals are supportive.")
            if review_strength == "above_market":
                parts.append("That rating edge is backed by a deeper review base.")
            elif review_strength == "below_market":
                parts.append("The review base is thinner than the comp set, so the increase stays moderate.")
            return " ".join(parts)
        if price_action == "lower":
            parts: list[str] = []
            if price_gap is not None and price_gap >= 3.0:
                parts.append("You are priced above the local comp average.")
            if quality_gap <= -0.05:
                parts.append("Review quality is below the market baseline.")
            if market_stats["market_pressure"] == "soft":
                parts.append("Demand signals are soft for the selected horizon.")
            if review_strength == "above_market":
                parts.append("Strong review depth softens the size of the reduction.")
            return " ".join(parts) or "The comp set does not support a higher rate right now."
        if price_action == "hold":
            return "The market signals are mixed, so the safest recommendation is to hold the current rate."
        return "Current price data is incomplete, so this is a directional benchmark rather than a firm recommendation."

    def _build_risk_note(
        self,
        *,
        price_stats: dict[str, Any],
        market_stats: dict[str, Any],
        review_volume_stats: dict[str, Any],
        confidence: Literal["low", "medium", "high"],
    ) -> str | None:
        notes: list[str] = []
        if confidence == "low":
            notes.append("Confidence is low because comp or market coverage is thin.")
        if price_stats["usable_neighbor_prices"] < 3:
            notes.append("Few usable neighbor prices were available.")
        if market_stats["weather_status"] == "skipped" or market_stats["events_status"] in {"error", "skipped"}:
            notes.append("Market-signal coverage is incomplete for this run.")
        if review_volume_stats["review_volume_strength"] == "below_market":
            notes.append("Your rating edge is supported by a thinner review base than nearby comps.")
        return " ".join(notes[:2]) or None

    def _build_narrative(
        self,
        *,
        prompt: str,
        recommendation: PricingRecommendation,
        signals: PricingSignalSummary,
    ) -> str:
        compact_summary = self._build_compact_summary(recommendation=recommendation, signals=signals)
        if not self._chat.is_available:
            text = self._deterministic_narrative(recommendation=recommendation)
            self._last_answer_step = StepLog(
                module="pricing_agent.answer_generation",
                prompt={"model": None, "fallback": True},
                response={"text": text, "fallback_used": True},
            )
            return text

        system_prompt = (
            "You are a hospitality pricing analyst. Explain a nightly-rate recommendation "
            "using only the provided computed facts. Do not invent booking data or conversion claims."
        )
        user_prompt = (
            f"User request: {prompt}\n\nComputed pricing summary:\n{compact_summary}\n\n"
            "Return at most 2 short paragraphs. Mention review volume only if it materially changes confidence or magnitude."
        )
        try:
            text = self._chat.generate(system_prompt=system_prompt, user_prompt=user_prompt)
        except Exception as exc:
            fallback = self._deterministic_narrative(recommendation=recommendation)
            self._last_answer_step = StepLog(
                module="pricing_agent.answer_generation",
                prompt={"model": self._chat.model, "system_prompt": system_prompt, "user_prompt": user_prompt},
                response={"error": f"{type(exc).__name__}: {exc}", "fallback_used": True, "text": fallback},
            )
            return fallback
        text = text.strip() or self._deterministic_narrative(recommendation=recommendation)
        self._last_answer_step = StepLog(
            module="pricing_agent.answer_generation",
            prompt={"model": self._chat.model, "system_prompt": system_prompt, "user_prompt": user_prompt},
            response={"text": text},
        )
        return text

    def _build_compact_summary(
        self,
        *,
        recommendation: PricingRecommendation,
        signals: PricingSignalSummary,
    ) -> str:
        lines = [
            f"Current price: {recommendation.current_price}",
            f"Recommended price: {recommendation.recommended_price}",
            f"Price change pct: {recommendation.price_change_pct}",
            f"Price action: {recommendation.price_action}",
            f"Confidence: {recommendation.confidence}",
            f"Primary reason: {recommendation.primary_reason}",
            f"Risk note: {recommendation.risk_note or 'none'}",
            f"Neighbor avg/min/max: {signals.neighbor_avg_price}/{signals.neighbor_min_price}/{signals.neighbor_max_price}",
            f"Review score gap: {signals.review_score_gap}",
            f"Review volume strength: {signals.review_volume_strength}",
            f"Owner total reviews: {signals.owner_number_of_reviews}",
            f"Neighbor avg total reviews: {signals.neighbor_avg_number_of_reviews}",
            f"Owner recent reviews 30d: {signals.owner_recent_reviews_30d}",
            f"Market pressure: {signals.market_pressure}",
            f"Demand signal count: {signals.demand_signal_count}",
        ]
        return "\n".join(lines)

    def _deterministic_narrative(self, *, recommendation: PricingRecommendation) -> str:
        current_price_text = (
            f"${recommendation.current_price:.2f}"
            if recommendation.current_price is not None
            else "an unknown current price"
        )
        recommended_text = (
            f"${recommendation.recommended_price:.2f}"
            if recommendation.recommended_price is not None
            else "the local comp average"
        )
        delta_text = (
            f"{recommendation.price_change_pct:+.1f}%"
            if recommendation.price_change_pct is not None
            else "n/a"
        )
        text = (
            f"{recommendation.price_action.title()} nightly price from {current_price_text} "
            f"to about {recommended_text} ({delta_text}). "
            f"Confidence: {recommendation.confidence}. {recommendation.primary_reason}"
        )
        if recommendation.risk_note:
            text += f" {recommendation.risk_note}"
        return text

    def _normalize_numeric(self, column: str, value: Any) -> float | None:
        if value is None:
            return None
        if isinstance(value, bool):
            return float(int(value))
        if isinstance(value, (int, float)):
            if isinstance(value, float) and math.isnan(value):
                return None
            return float(value)
        raw = str(value).strip()
        if not raw:
            return None
        raw = raw.replace(",", "")
        if column == "price":
            raw = raw.replace("$", "")
        if raw.endswith("%"):
            raw = raw[:-1]
        match = re.search(r"-?\d+(?:\.\d+)?", raw)
        if not match:
            return None
        try:
            return float(match.group(0))
        except ValueError:
            return None

    def _normalize_int(self, value: Any) -> int | None:
        numeric = self._normalize_float(value)
        if numeric is None:
            return None
        return int(round(numeric))

    def _normalize_float(self, value: Any) -> float | None:
        return self._normalize_numeric("number", value)

    def _recent_review_momentum(self, row: dict[str, Any]) -> float | None:
        recent_30d = self._normalize_float(row.get("number_of_reviews_l30d"))
        if recent_30d is not None:
            return recent_30d
        reviews_per_month = self._normalize_float(row.get("reviews_per_month"))
        if reviews_per_month is not None:
            return reviews_per_month
        reviews_ltm = self._normalize_float(row.get("number_of_reviews_ltm"))
        if reviews_ltm is not None:
            return reviews_ltm / 12.0
        return None

    def _ratio(self, owner_value: float | int | None, neighbor_avg: float | None) -> float | None:
        if owner_value is None or neighbor_avg is None or neighbor_avg <= 0:
            return None
        return float(owner_value) / neighbor_avg


def outcome_to_response(outcome: PricingRunOutcome) -> PricingResponse:
    """Convert internal outcome into the public structured pricing response."""

    return PricingResponse(
        status=("error" if outcome.error else "ok"),
        error=outcome.error,
        response=(None if outcome.error else outcome.narrative),
        recommendation=outcome.recommendation,
        signals=outcome.signals,
        steps=outcome.steps,
    )
