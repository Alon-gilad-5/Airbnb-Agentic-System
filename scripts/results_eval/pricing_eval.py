"""Offline evaluator for pricing-agent recommendation logic."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
import time
from typing import Any

from app.agents.pricing_agent import PricingAgent, PricingAgentConfig
from app.schemas import StepLog
from app.services.market_data_providers import NearbyEvent, PublicHoliday, WeatherForecastDay
from scripts.results_eval.common import compute_reliability, compute_task_success_rate, parse_cases


@dataclass
class _DummyChatService:
    is_available: bool = False
    model: str = "offline-eval"

    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        raise RuntimeError("Chat unavailable in offline evaluation mode.")


class _DummyNeighborStore:
    def __init__(self, neighbors: list[str] | None = None) -> None:
        self._neighbors = neighbors or []

    def get_neighbors(self, property_id: str) -> list[str] | None:
        return list(self._neighbors)


class _DummyListingStore:
    def __init__(self, rows: list[dict[str, Any]] | None = None) -> None:
        self._rows = rows or []

    def get_listings_by_ids(self, listing_ids: list[str], columns: list[str]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for row in self._rows:
            if str(row.get("id")) not in listing_ids:
                continue
            out.append({column: row.get(column) for column in ["id", "name", *columns]})
        return out


class _DummyProviders:
    def __init__(
        self,
        *,
        weather: list[WeatherForecastDay] | None = None,
        events: list[NearbyEvent] | None = None,
        holidays: list[PublicHoliday] | None = None,
    ) -> None:
        self._weather = weather or []
        self._events = events or []
        self._holidays = holidays or []

    def fetch_weather_forecast(self, **kwargs: Any) -> tuple[list[WeatherForecastDay], dict[str, Any]]:
        return self._weather, {"status": "ok", "provider": "test-weather"}

    def fetch_ticketmaster_events(self, **kwargs: Any) -> tuple[list[NearbyEvent], dict[str, Any]]:
        return self._events, {"status": "ok", "provider": "test-events"}

    def fetch_us_public_holidays(self, **kwargs: Any) -> tuple[list[PublicHoliday], dict[str, Any]]:
        return self._holidays, {"status": "ok", "provider": "test-holidays"}


VALID_MODULES = {
    "pricing_agent.context_resolve",
    "pricing_agent.neighbor_lookup",
    "pricing_agent.data_fetch",
    "pricing_agent.market_signal_fetch",
    "pricing_agent.recommendation_compute",
    "pricing_agent.answer_generation",
}


def _is_step_log(step: Any) -> bool:
    return isinstance(step, StepLog) and isinstance(step.module, str) and isinstance(step.prompt, dict) and isinstance(step.response, dict)


def _to_weather(rows: list[dict[str, Any]]) -> list[WeatherForecastDay]:
    out: list[WeatherForecastDay] = []
    for row in rows:
        out.append(
            WeatherForecastDay(
                day=datetime.fromisoformat(str(row["day"])).date(),
                weather_code=int(row["weather_code"]) if row.get("weather_code") is not None else None,
                temp_max_c=float(row["temp_max_c"]) if row.get("temp_max_c") is not None else None,
                temp_min_c=float(row["temp_min_c"]) if row.get("temp_min_c") is not None else None,
                precipitation_mm=float(row["precipitation_mm"]) if row.get("precipitation_mm") is not None else None,
                snowfall_cm=float(row["snowfall_cm"]) if row.get("snowfall_cm") is not None else None,
                wind_kph_max=float(row["wind_kph_max"]) if row.get("wind_kph_max") is not None else None,
            )
        )
    return out


def _to_events(rows: list[dict[str, Any]]) -> list[NearbyEvent]:
    out: list[NearbyEvent] = []
    for row in rows:
        raw_start = row.get("start_at_utc")
        start_at = datetime.fromisoformat(str(raw_start)).astimezone(UTC) if raw_start else None
        out.append(
            NearbyEvent(
                name=str(row.get("name", "Event")),
                source_url=row.get("source_url"),
                start_at_utc=start_at,
                venue_name=row.get("venue_name"),
                latitude=float(row["latitude"]) if row.get("latitude") is not None else None,
                longitude=float(row["longitude"]) if row.get("longitude") is not None else None,
                distance_km=float(row["distance_km"]) if row.get("distance_km") is not None else None,
                category=row.get("category"),
                popularity_hint=row.get("popularity_hint"),
            )
        )
    return out


def _to_holidays(rows: list[dict[str, Any]]) -> list[PublicHoliday]:
    out: list[PublicHoliday] = []
    for row in rows:
        out.append(
            PublicHoliday(
                day=datetime.fromisoformat(str(row["day"])).date(),
                local_name=str(row.get("local_name", row.get("name", "Holiday"))),
                name=str(row.get("name", "Holiday")),
            )
        )
    return out


def evaluate_pricing(
    *,
    repo_root: Path,
    split: str,
) -> dict[str, Any]:
    """Evaluate pricing recommendations against deterministic oracle cases."""

    cases = parse_cases(repo_root / "eval" / "cases" / "pricing_cases.jsonl", split=split)
    case_results: list[dict[str, Any]] = []
    direction_hits = 0
    direction_total = 0

    for case in cases:
        started = time.perf_counter()
        exception = None
        contract_ok = False
        trace_ok = False
        passed = False
        failure_reason = "uninitialized"
        try:
            fixtures = case.raw.get("fixtures") if isinstance(case.raw.get("fixtures"), dict) else {}
            agent = PricingAgent(
                listing_store=_DummyListingStore(rows=fixtures.get("rows", [])),
                neighbor_store=_DummyNeighborStore(neighbors=fixtures.get("neighbors", [])),
                market_data_providers=_DummyProviders(
                    weather=_to_weather(fixtures.get("weather", [])),
                    events=_to_events(fixtures.get("events", [])),
                    holidays=_to_holidays(fixtures.get("holidays", [])),
                ),
                chat_service=_DummyChatService(),
                config=PricingAgentConfig(),
            )
            outcome = agent.recommend(case.prompt, context=case.context)
            modules = [step.module for step in outcome.steps]
            contract_ok = (
                isinstance(outcome.narrative, str)
                and isinstance(outcome.error, (str, type(None)))
                and all(_is_step_log(step) for step in outcome.steps)
            )
            required_modules = set(case.expected.get("required_modules", ["pricing_agent.answer_generation"]))
            trace_ok = set(modules).issubset(VALID_MODULES) and required_modules.issubset(set(modules))

            expected_status = str(case.expected.get("status", "ok"))
            if expected_status == "error":
                error_contains = str(case.expected.get("error_contains", "")).strip().lower()
                passed = bool(outcome.error) and (error_contains in outcome.error.lower())
                failure_reason = None if passed else "error_expectation_mismatch"
            else:
                direction_total += 1
                recommendation = outcome.recommendation
                signals = outcome.signals
                if recommendation is None or signals is None:
                    passed = False
                    failure_reason = "missing_recommendation_or_signals"
                else:
                    direction_ok = recommendation.price_action == case.expected.get("price_action")
                    direction_hits += 1 if direction_ok else 0
                    confidence_ok = (
                        ("confidence" not in case.expected)
                        or (recommendation.confidence == case.expected.get("confidence"))
                    )
                    pct_ok = True
                    if "price_change_pct_min" in case.expected:
                        pct_ok = pct_ok and (
                            recommendation.price_change_pct is not None
                            and float(recommendation.price_change_pct) >= float(case.expected["price_change_pct_min"])
                        )
                    if "price_change_pct_max" in case.expected:
                        pct_ok = pct_ok and (
                            recommendation.price_change_pct is not None
                            and float(recommendation.price_change_pct) <= float(case.expected["price_change_pct_max"])
                        )
                    pressure_ok = (
                        ("market_pressure" not in case.expected)
                        or (signals.market_pressure == case.expected.get("market_pressure"))
                    )
                    passed = outcome.error is None and direction_ok and confidence_ok and pct_ok and pressure_ok
                    failure_reason = None if passed else "recommendation_expectation_mismatch"
        except Exception as exc:  # pragma: no cover - defensive guard
            exception = f"{type(exc).__name__}: {exc}"
            failure_reason = "exception"

        latency_ms = (time.perf_counter() - started) * 1000.0
        case_results.append(
            {
                "agent": "pricing",
                "split": split,
                "case_id": case.case_id,
                "pass": passed,
                "failure_reason": failure_reason,
                "latency_ms": round(latency_ms, 3),
                "metadata": {
                    "contract_ok": contract_ok,
                    "step_trace_ok": trace_ok,
                    "exception": exception,
                },
            }
        )

    reliability = compute_reliability(case_results)
    direction_accuracy = (direction_hits / direction_total) if direction_total else 0.0
    task_success_rate = compute_task_success_rate(case_results)
    return {
        "agent": "pricing",
        "split": split,
        "primary_metric_name": "recommendation_direction_accuracy",
        "primary_metric": round(direction_accuracy, 4),
        "metrics": {
            "case_count": len(case_results),
            "recommendation_direction_accuracy": round(direction_accuracy, 4),
            "task_success_rate": task_success_rate,
        },
        "reliability": reliability,
        "task_success_rate": task_success_rate,
        "case_results": case_results,
    }

