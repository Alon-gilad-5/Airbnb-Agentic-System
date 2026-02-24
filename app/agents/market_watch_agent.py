"""Autonomous market-watch agent for events, weather, and demand signals.

Uses a LangChain-first architecture with composable RunnableLambda stages
for the market-watch pipeline (zero LLM calls -- fully deterministic).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any, TypedDict

from langchain_core.runnables import RunnableLambda

from app.agents.base import Agent, AgentResult
from app.schemas import StepLog
from app.services.market_alert_store import (
    MarketAlertRecord,
    MarketAlertStore,
    build_alert_id,
    utc_now_iso,
)
from app.services.market_data_providers import (
    MarketDataProviders,
    NearbyEvent,
    PublicHoliday,
    WeatherForecastDay,
    utc_now,
    within_days,
)

NO_EVIDENCE_RESPONSE = "I couldn't find enough data to answer your question."


class AlertCandidate(TypedDict):
    alert_type: str
    severity: str
    title: str
    summary: str
    start_at_utc: str | None
    end_at_utc: str | None
    source_name: str
    source_url: str | None
    evidence: dict[str, Any]


@dataclass
class MarketWatchAgentConfig:
    """Runtime knobs controlling signal horizon, thresholds, and output size."""

    lookahead_days: int = 14
    event_radius_km: int = 15
    max_alerts_per_run: int = 8
    storm_wind_kph_threshold: float = 45.0
    heavy_rain_mm_threshold: float = 20.0
    snow_cm_threshold: float = 4.0
    max_answer_words: int = 170


@dataclass
class MarketWatchRunOutcome:
    """Internal result that includes persisted alert details for special endpoints."""

    response: str
    steps: list[StepLog]
    alerts: list[MarketAlertRecord]
    inserted_count: int


# ---------------------------------------------------------------------------
# Module-level helper functions
# ---------------------------------------------------------------------------


def _context_str(ctx: dict[str, Any], key: str) -> str | None:
    value = ctx.get(key)
    if isinstance(value, str):
        trimmed = value.strip()
        return trimmed or None
    return None


def _context_float(ctx: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = ctx.get(key)
        if value is None:
            continue
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            text = value.strip()
            if not text:
                continue
            try:
                return float(text)
            except ValueError:
                continue
        else:
            continue
    return None


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_km = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(delta_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    )
    return radius_km * (2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)))


def _category_weight(category: str | None) -> int:
    label = (category or "").strip().lower()
    if any(token in label for token in ["music", "concert", "sports", "festival"]):
        return 2
    if any(token in label for token in ["theatre", "family", "film", "conference"]):
        return 1
    return 0


def _start_of_day_utc(day_value: date) -> str:
    return datetime(day_value.year, day_value.month, day_value.day, tzinfo=UTC).isoformat()


def _end_of_day_utc(day_value: date) -> str:
    return datetime(day_value.year, day_value.month, day_value.day, 23, 59, 59, tzinfo=UTC).isoformat()


def _days_delta(days: int) -> timedelta:
    return timedelta(days=max(0, int(days)))


def _cap_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + " ..."


def _attach_event_distances(
    events: list[NearbyEvent],
    center_lat: float,
    center_lon: float,
) -> list[NearbyEvent]:
    out: list[NearbyEvent] = []
    for event in events:
        if event.latitude is not None and event.longitude is not None:
            event.distance_km = _haversine_km(center_lat, center_lon, event.latitude, event.longitude)
        out.append(event)
    return out


def _to_alert_candidate(
    *,
    alert_type: str,
    severity: str,
    title: str,
    summary: str,
    start_at_utc: str | None,
    end_at_utc: str | None,
    source_name: str,
    source_url: str | None,
    evidence: dict[str, Any],
) -> AlertCandidate:
    return AlertCandidate(
        alert_type=alert_type,
        severity=severity,
        title=title,
        summary=summary,
        start_at_utc=start_at_utc,
        end_at_utc=end_at_utc,
        source_name=source_name,
        source_url=source_url,
        evidence=evidence,
    )


# ---------------------------------------------------------------------------
# LangChain-first market-watch pipeline
# ---------------------------------------------------------------------------


class MarketWatchPipeline:
    """LangChain-first market-watch pipeline composed of RunnableLambda stages.

    Fully deterministic (zero LLM calls). Each stage accepts pipeline state
    and returns incremental updates. The orchestrator merges updates and
    handles the single conditional branch (missing coordinates -> early exit).
    """

    def __init__(
        self,
        *,
        providers: MarketDataProviders,
        alert_store: MarketAlertStore,
        config: MarketWatchAgentConfig,
    ) -> None:
        self._providers = providers
        self._alert_store = alert_store
        self._config = config

        self.extract_context = RunnableLambda(self._extract_context_stage).with_config(
            run_name="ExtractContext",
        )
        self.fetch_signals = RunnableLambda(self._fetch_signals_stage).with_config(
            run_name="FetchSignals",
        )
        self.analyze_weather = RunnableLambda(self._analyze_weather_stage).with_config(
            run_name="AnalyzeWeather",
        )
        self.analyze_events = RunnableLambda(self._analyze_events_stage).with_config(
            run_name="AnalyzeEvents",
        )
        self.analyze_demand = RunnableLambda(self._analyze_demand_stage).with_config(
            run_name="AnalyzeDemand",
        )
        self.select_alerts = RunnableLambda(self._select_alerts_stage).with_config(
            run_name="SelectAlerts",
        )
        self.persist_alerts = RunnableLambda(self._persist_alerts_stage).with_config(
            run_name="PersistAlerts",
        )
        self.build_answer = RunnableLambda(self._build_answer_stage).with_config(
            run_name="BuildAnswer",
        )

    # -- state merge helper --

    @staticmethod
    def _apply(state: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
        """Merge stage updates into pipeline state, accumulating steps via addition."""
        merged = dict(state)
        new_steps = updates.pop("steps", [])
        merged.update(updates)
        merged["steps"] = merged.get("steps", []) + new_steps
        return merged

    # -- main orchestration --

    def invoke(self, state: dict[str, Any]) -> dict[str, Any]:
        """Execute the full market-watch pipeline with conditional branching."""

        state = self._apply(state, self.extract_context.invoke(state))

        if not state.get("has_coordinates"):
            return state

        state = self._apply(state, self.fetch_signals.invoke(state))
        state = self._apply(state, self.analyze_weather.invoke(state))
        state = self._apply(state, self.analyze_events.invoke(state))
        state = self._apply(state, self.analyze_demand.invoke(state))
        state = self._apply(state, self.select_alerts.invoke(state))
        state = self._apply(state, self.persist_alerts.invoke(state))

        if state.get("answer") == NO_EVIDENCE_RESPONSE:
            return state

        state = self._apply(state, self.build_answer.invoke(state))
        return state

    # -- stage implementations --

    def _extract_context_stage(self, state: dict[str, Any]) -> dict[str, Any]:
        ctx = state.get("context") or {}
        cfg = self._config
        base_context: dict[str, str | None] = {
            "owner_id": _context_str(ctx, "owner_id"),
            "owner_name": _context_str(ctx, "owner_name"),
            "property_id": _context_str(ctx, "property_id"),
            "property_name": _context_str(ctx, "property_name"),
            "city": _context_str(ctx, "city"),
            "region": _context_str(ctx, "region"),
        }
        latitude = _context_float(ctx, "latitude", "lat")
        longitude = _context_float(ctx, "longitude", "lon")
        has_coordinates = latitude is not None and longitude is not None

        result: dict[str, Any] = {
            "base_context": base_context,
            "latitude": latitude,
            "longitude": longitude,
            "has_coordinates": has_coordinates,
        }

        if not has_coordinates:
            result["answer"] = NO_EVIDENCE_RESPONSE
            result["steps"] = [
                StepLog(
                    module="market_watch_agent.signal_collection",
                    prompt={"lookahead_days": cfg.lookahead_days},
                    response={
                        "status": "missing_coordinates",
                        "required": ["ACTIVE_PROPERTY_LAT", "ACTIVE_PROPERTY_LON"],
                    },
                ),
                StepLog(
                    module="market_watch_agent.answer_generation",
                    prompt={"mode": "insufficient_data"},
                    response={"text": NO_EVIDENCE_RESPONSE},
                ),
            ]

        return result

    def _fetch_signals_stage(self, state: dict[str, Any]) -> dict[str, Any]:
        latitude = state["latitude"]
        longitude = state["longitude"]
        prompt = state.get("prompt", "")
        cfg = self._config

        try:
            weather, weather_meta = self._providers.fetch_weather_forecast(
                latitude=latitude,
                longitude=longitude,
                lookahead_days=cfg.lookahead_days,
            )

            now = utc_now()
            events, events_meta = self._providers.fetch_ticketmaster_events(
                latitude=latitude,
                longitude=longitude,
                radius_km=cfg.event_radius_km,
                start_at_utc=now,
                end_at_utc=now.replace(microsecond=0) + _days_delta(cfg.lookahead_days),
            )
            events = _attach_event_distances(events, latitude, longitude)

            holiday_years = {now.year, (now + _days_delta(cfg.lookahead_days)).year}
            holidays: list[PublicHoliday] = []
            holiday_meta: list[dict[str, Any]] = []
            for year in sorted(holiday_years):
                items, meta = self._providers.fetch_us_public_holidays(year=year)
                holidays.extend(items)
                holiday_meta.append(meta)
            holidays = [h for h in holidays if within_days(now, h.day, cfg.lookahead_days)]

            return {
                "weather": weather,
                "events": events,
                "holidays": holidays,
                "steps": [
                    StepLog(
                        module="market_watch_agent.signal_collection",
                        prompt={
                            "lookahead_days": cfg.lookahead_days,
                            "event_radius_km": cfg.event_radius_km,
                            "prompt": prompt,
                        },
                        response={
                            "weather_provider": weather_meta,
                            "events_provider": events_meta,
                            "holiday_providers": holiday_meta,
                            "weather_count": len(weather),
                            "event_count": len(events),
                            "holiday_count": len(holidays),
                        },
                    )
                ],
            }
        except Exception as exc:
            return {
                "weather": [],
                "events": [],
                "holidays": [],
                "steps": [
                    StepLog(
                        module="market_watch_agent.signal_collection",
                        prompt={
                            "lookahead_days": cfg.lookahead_days,
                            "event_radius_km": cfg.event_radius_km,
                            "prompt": prompt,
                        },
                        response={"error": f"{type(exc).__name__}: {exc}"},
                    )
                ],
            }

    def _analyze_weather_stage(self, state: dict[str, Any]) -> dict[str, Any]:
        weather = state.get("weather") or []
        base_context = state.get("base_context") or {}
        cfg = self._config
        candidates: list[AlertCandidate] = []

        for day in weather:
            if day.wind_kph_max is not None and day.wind_kph_max >= cfg.storm_wind_kph_threshold:
                candidates.append(_to_alert_candidate(
                    alert_type="weather", severity="high",
                    title=f"Storm-level wind risk on {day.day.isoformat()}",
                    summary=(
                        f"Forecast max wind is {day.wind_kph_max:.1f} kph. "
                        "Prepare proactive guest communication and check external fixtures."
                    ),
                    start_at_utc=_start_of_day_utc(day.day),
                    end_at_utc=_end_of_day_utc(day.day),
                    source_name="Open-Meteo", source_url="https://open-meteo.com/",
                    evidence={
                        "wind_kph_max": day.wind_kph_max,
                        "precipitation_mm": day.precipitation_mm,
                        "snowfall_cm": day.snowfall_cm,
                        "property_id": base_context.get("property_id"),
                    },
                ))
            if day.precipitation_mm is not None and day.precipitation_mm >= cfg.heavy_rain_mm_threshold:
                candidates.append(_to_alert_candidate(
                    alert_type="operations", severity="medium",
                    title=f"Heavy rain expected on {day.day.isoformat()}",
                    summary=(
                        f"Forecast rain is {day.precipitation_mm:.1f} mm. "
                        "Review check-in logistics, parking guidance, and maintenance readiness."
                    ),
                    start_at_utc=_start_of_day_utc(day.day),
                    end_at_utc=_end_of_day_utc(day.day),
                    source_name="Open-Meteo", source_url="https://open-meteo.com/",
                    evidence={
                        "wind_kph_max": day.wind_kph_max,
                        "precipitation_mm": day.precipitation_mm,
                        "snowfall_cm": day.snowfall_cm,
                        "property_id": base_context.get("property_id"),
                    },
                ))
            if day.snowfall_cm is not None and day.snowfall_cm >= cfg.snow_cm_threshold:
                candidates.append(_to_alert_candidate(
                    alert_type="operations", severity="high",
                    title=f"Snow disruption risk on {day.day.isoformat()}",
                    summary=(
                        f"Forecast snowfall is {day.snowfall_cm:.1f} cm. "
                        "Plan access instructions and early guest messaging for travel disruptions."
                    ),
                    start_at_utc=_start_of_day_utc(day.day),
                    end_at_utc=_end_of_day_utc(day.day),
                    source_name="Open-Meteo", source_url="https://open-meteo.com/",
                    evidence={
                        "wind_kph_max": day.wind_kph_max,
                        "precipitation_mm": day.precipitation_mm,
                        "snowfall_cm": day.snowfall_cm,
                        "property_id": base_context.get("property_id"),
                    },
                ))

        return {
            "weather_alerts": candidates,
            "steps": [
                StepLog(
                    module="market_watch_agent.weather_analysis",
                    prompt={
                        "storm_wind_kph_threshold": cfg.storm_wind_kph_threshold,
                        "heavy_rain_mm_threshold": cfg.heavy_rain_mm_threshold,
                        "snow_cm_threshold": cfg.snow_cm_threshold,
                    },
                    response={
                        "forecast_days": len(weather),
                        "candidate_count": len(candidates),
                    },
                )
            ],
        }

    def _analyze_events_stage(self, state: dict[str, Any]) -> dict[str, Any]:
        events = state.get("events") or []
        base_context = state.get("base_context") or {}
        cfg = self._config
        candidates: list[AlertCandidate] = []
        scored_events: list[dict[str, Any]] = []
        now = utc_now()

        for event in events:
            if event.start_at_utc is None or event.distance_km is None:
                continue
            if event.distance_km > float(cfg.event_radius_km):
                continue

            days_until = max(0, (event.start_at_utc.date() - now.date()).days)
            cat_weight = _category_weight(event.category)
            proximity_bonus = 1 if event.distance_km <= 5 else 0
            urgency_bonus = 1 if days_until <= 3 else 0
            popularity_bonus = 1 if (event.popularity_hint or "").lower() == "high" else 0
            impact_score = cat_weight + proximity_bonus + urgency_bonus + popularity_bonus
            severity = "high" if impact_score >= 4 else "medium" if impact_score >= 3 else "low"

            scored_events.append({
                "name": event.name,
                "distance_km": round(event.distance_km, 2),
                "days_until": days_until,
                "category": event.category,
                "impact_score": impact_score,
                "severity": severity,
            })

            if severity == "low":
                continue

            candidates.append(_to_alert_candidate(
                alert_type="event", severity=severity,
                title=f"Nearby {severity}-impact event: {event.name}",
                summary=(
                    f"{event.name} is about {event.distance_km:.1f} km away in {days_until} day(s). "
                    "Consider price and staffing adjustments for possible demand lift."
                ),
                start_at_utc=event.start_at_utc.astimezone(UTC).isoformat(),
                end_at_utc=None,
                source_name="Ticketmaster", source_url=event.source_url,
                evidence={
                    "event_name": event.name,
                    "category": event.category,
                    "distance_km": event.distance_km,
                    "impact_score": impact_score,
                    "property_id": base_context.get("property_id"),
                },
            ))

        return {
            "event_alerts": candidates,
            "scored_events": scored_events,
            "steps": [
                StepLog(
                    module="market_watch_agent.event_analysis",
                    prompt={"event_radius_km": cfg.event_radius_km},
                    response={
                        "raw_events_count": len(events),
                        "scored_events_count": len(scored_events),
                        "candidate_count": len(candidates),
                    },
                )
            ],
        }

    def _analyze_demand_stage(self, state: dict[str, Any]) -> dict[str, Any]:
        holidays = state.get("holidays") or []
        scored_events = state.get("scored_events") or []
        base_context = state.get("base_context") or {}
        cfg = self._config
        candidates: list[AlertCandidate] = []
        now = utc_now()

        upcoming_holidays = sorted(
            [h for h in holidays if within_days(now, h.day, cfg.lookahead_days)],
            key=lambda h: h.day,
        )
        medium_plus_events = [e for e in scored_events if str(e.get("severity")) in {"medium", "high"}]

        if upcoming_holidays or len(medium_plus_events) >= 2:
            holiday_label = upcoming_holidays[0].name if upcoming_holidays else "upcoming period"
            severity = "high" if (upcoming_holidays and len(medium_plus_events) >= 2) else "medium"
            candidates.append(_to_alert_candidate(
                alert_type="demand", severity=severity,
                title=f"Demand opportunity signal around {holiday_label}",
                summary=(
                    f"Detected {len(medium_plus_events)} medium/high nearby events "
                    f"and {len(upcoming_holidays)} holiday signal(s) in the next {cfg.lookahead_days} days. "
                    "Consider testing higher nightly rates and minimum-stay strategy."
                ),
                start_at_utc=_start_of_day_utc(upcoming_holidays[0].day) if upcoming_holidays else None,
                end_at_utc=None,
                source_name="Nager.Date + Ticketmaster",
                source_url="https://date.nager.at/",
                evidence={
                    "holiday_count": len(upcoming_holidays),
                    "event_signal_count": len(medium_plus_events),
                    "property_id": base_context.get("property_id"),
                },
            ))

        return {
            "demand_alerts": candidates,
            "steps": [
                StepLog(
                    module="market_watch_agent.demand_analysis",
                    prompt={"lookahead_days": cfg.lookahead_days},
                    response={
                        "holiday_count": len(upcoming_holidays),
                        "event_signal_count": len(medium_plus_events),
                        "candidate_count": len(candidates),
                    },
                )
            ],
        }

    def _select_alerts_stage(self, state: dict[str, Any]) -> dict[str, Any]:
        weather_alerts = state.get("weather_alerts") or []
        event_alerts = state.get("event_alerts") or []
        demand_alerts = state.get("demand_alerts") or []
        candidates = weather_alerts + event_alerts + demand_alerts

        severity_rank = {"high": 0, "medium": 1, "low": 2}
        ordered = sorted(
            candidates,
            key=lambda c: (
                severity_rank.get(c["severity"], 9),
                c.get("start_at_utc") or "9999-12-31T23:59:59+00:00",
            ),
        )
        selected = ordered[:self._config.max_alerts_per_run]

        return {
            "selected": selected,
            "steps": [
                StepLog(
                    module="market_watch_agent.alert_decision",
                    prompt={"max_alerts_per_run": self._config.max_alerts_per_run},
                    response={
                        "candidate_count": len(candidates),
                        "selected_count": len(selected),
                        "selected_titles": [c["title"] for c in selected[:6]],
                    },
                )
            ],
        }

    def _persist_alerts_stage(self, state: dict[str, Any]) -> dict[str, Any]:
        selected = state.get("selected") or []
        base_context = state.get("base_context") or {}
        persist = state.get("persist_alerts", False)
        weather = state.get("weather") or []
        events = state.get("events") or []
        holidays = state.get("holidays") or []

        insufficient_data = not weather and not events and not holidays and not selected
        if insufficient_data:
            return {
                "records": [],
                "inserted_count": 0,
                "answer": NO_EVIDENCE_RESPONSE,
                "steps": [
                    StepLog(
                        module="market_watch_agent.inbox_write",
                        prompt={"persist_alerts": persist, "attempted": 0},
                        response={"status": "skipped", "reason": "insufficient_data"},
                    ),
                    StepLog(
                        module="market_watch_agent.answer_generation",
                        prompt={"mode": "insufficient_data"},
                        response={"text": NO_EVIDENCE_RESPONSE},
                    ),
                ],
            }

        created_at = utc_now_iso()
        records: list[MarketAlertRecord] = []
        for candidate in selected:
            records.append(
                MarketAlertRecord(
                    id=build_alert_id(
                        owner_id=base_context.get("owner_id"),
                        property_id=base_context.get("property_id"),
                        alert_type=candidate["alert_type"],
                        title=candidate["title"],
                        start_at_utc=candidate.get("start_at_utc"),
                    ),
                    created_at_utc=created_at,
                    owner_id=base_context.get("owner_id"),
                    property_id=base_context.get("property_id"),
                    property_name=base_context.get("property_name"),
                    city=base_context.get("city"),
                    region=base_context.get("region"),
                    alert_type=candidate["alert_type"],
                    severity=candidate["severity"],
                    title=candidate["title"],
                    summary=candidate["summary"],
                    start_at_utc=candidate.get("start_at_utc"),
                    end_at_utc=candidate.get("end_at_utc"),
                    source_name=candidate["source_name"],
                    source_url=candidate.get("source_url"),
                    evidence=candidate["evidence"],
                )
            )

        inserted_count = 0
        if persist and records:
            inserted_count = self._alert_store.insert_alerts(records)
            step = StepLog(
                module="market_watch_agent.inbox_write",
                prompt={"persist_alerts": True, "attempted": len(records)},
                response={"status": "ok", "inserted": inserted_count},
            )
        elif persist:
            step = StepLog(
                module="market_watch_agent.inbox_write",
                prompt={"persist_alerts": True, "attempted": 0},
                response={"status": "skipped", "reason": "no_alerts_to_persist"},
            )
        else:
            step = StepLog(
                module="market_watch_agent.inbox_write",
                prompt={"persist_alerts": False, "attempted": len(records)},
                response={"status": "skipped", "reason": "on_demand_mode"},
            )

        return {"records": records, "inserted_count": inserted_count, "steps": [step]}

    def _build_answer_stage(self, state: dict[str, Any]) -> dict[str, Any]:
        records = state.get("records") or []
        cfg = self._config

        if not records:
            answer = (
                f"No notable market-watch signals were detected for the next {cfg.lookahead_days} days "
                "based on current event, weather, and holiday data."
            )
        else:
            lines: list[str] = [
                f"Detected {len(records)} actionable market signal(s) for the next {cfg.lookahead_days} days:",
            ]
            citations: list[str] = []
            for idx, alert in enumerate(records[:5], start=1):
                when = f" (starts {alert.start_at_utc})" if alert.start_at_utc else ""
                lines.append(f"{idx}. [{alert.severity}] {alert.title}{when}: {alert.summary}")
                citation = f"{alert.source_name}{' + ' + alert.source_url if alert.source_url else ''}"
                if citation not in citations:
                    citations.append(citation)
            if citations:
                lines.append("")
                lines.append("Sources:")
                lines.extend([f"- {citation}" for citation in citations[:5]])
            answer = "\n".join(lines)

        answer = _cap_words(answer, cfg.max_answer_words)

        return {
            "answer": answer,
            "steps": [
                StepLog(
                    module="market_watch_agent.answer_generation",
                    prompt={"mode": "deterministic_template", "alert_count": len(records)},
                    response={"text": answer},
                )
            ],
        }


# ---------------------------------------------------------------------------
# MarketWatchAgent -- thin wrapper delegating to MarketWatchPipeline
# ---------------------------------------------------------------------------


class MarketWatchAgent(Agent):
    """Agent that can run on-demand or autonomously to produce proactive intelligence.

    Thin wrapper that delegates execution to a LangChain-first MarketWatchPipeline.
    """

    name = "market_watch_agent"

    def __init__(
        self,
        *,
        providers: MarketDataProviders,
        alert_store: MarketAlertStore,
        config: MarketWatchAgentConfig | None = None,
    ) -> None:
        self.providers = providers
        self.alert_store = alert_store
        self.config = config or MarketWatchAgentConfig()
        self._pipeline = MarketWatchPipeline(
            providers=providers,
            alert_store=alert_store,
            config=self.config,
        )

    def run(self, prompt: str, context: dict[str, object] | None = None) -> AgentResult:
        """Handle direct user prompts routed to market-watch without persisting alerts."""

        result = self._pipeline.invoke({
            "prompt": prompt,
            "context": context or {},
            "persist_alerts": False,
            "steps": [],
        })
        return AgentResult(
            response=result.get("answer", NO_EVIDENCE_RESPONSE),
            steps=result.get("steps", []),
        )

    def run_autonomous(self, context: dict[str, object] | None = None) -> MarketWatchRunOutcome:
        """Run autonomous cycle, persist selected alerts, and return full trace."""

        result = self._pipeline.invoke({
            "prompt": "autonomous market watch cycle",
            "context": context or {},
            "persist_alerts": True,
            "steps": [],
        })
        return MarketWatchRunOutcome(
            response=result.get("answer", NO_EVIDENCE_RESPONSE),
            steps=result.get("steps", []),
            alerts=result.get("records", []),
            inserted_count=result.get("inserted_count", 0),
        )
