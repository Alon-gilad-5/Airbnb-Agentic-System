"""Autonomous market-watch agent for events, weather, and demand signals."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
import math
from typing import Any

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
class AlertCandidate:
    """In-memory alert candidate before persistence."""

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
class MarketWatchRunOutcome:
    """Internal result that includes persisted alert details for special endpoints."""

    response: str
    steps: list[StepLog]
    alerts: list[MarketAlertRecord]
    inserted_count: int


class MarketWatchAgent(Agent):
    """Agent that can run on-demand or autonomously to produce proactive intelligence."""

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

    def run(self, prompt: str, context: dict[str, object] | None = None) -> AgentResult:
        """Handle direct user prompts routed to market-watch without persisting alerts."""

        outcome = self._execute(prompt=prompt, context=context or {}, persist_alerts=False)
        return AgentResult(response=outcome.response, steps=outcome.steps)

    def run_autonomous(self, context: dict[str, object] | None = None) -> MarketWatchRunOutcome:
        """Run autonomous cycle, persist selected alerts, and return full trace."""

        return self._execute(
            prompt="autonomous market watch cycle",
            context=context or {},
            persist_alerts=True,
        )

    def _execute(
        self,
        *,
        prompt: str,
        context: dict[str, object],
        persist_alerts: bool,
    ) -> MarketWatchRunOutcome:
        """Collect signals, generate alerts, optionally persist, and produce deterministic answer."""

        steps: list[StepLog] = []
        base_context = self._extract_base_context(context)
        latitude = self._context_float(context, "latitude", "lat")
        longitude = self._context_float(context, "longitude", "lon")

        if latitude is None or longitude is None:
            steps.append(
                StepLog(
                    module="market_watch_agent.signal_collection",
                    prompt={"lookahead_days": self.config.lookahead_days},
                    response={
                        "status": "missing_coordinates",
                        "required": ["ACTIVE_PROPERTY_LAT", "ACTIVE_PROPERTY_LON"],
                    },
                )
            )
            steps.append(
                StepLog(
                    module="market_watch_agent.answer_generation",
                    prompt={"mode": "insufficient_data"},
                    response={"text": NO_EVIDENCE_RESPONSE},
                )
            )
            return MarketWatchRunOutcome(response=NO_EVIDENCE_RESPONSE, steps=steps, alerts=[], inserted_count=0)

        weather, weather_meta = self.providers.fetch_weather_forecast(
            latitude=latitude,
            longitude=longitude,
            lookahead_days=self.config.lookahead_days,
        )

        now_utc = utc_now()
        events, events_meta = self.providers.fetch_ticketmaster_events(
            latitude=latitude,
            longitude=longitude,
            radius_km=self.config.event_radius_km,
            start_at_utc=now_utc,
            end_at_utc=now_utc.replace(microsecond=0) + self._days_delta(self.config.lookahead_days),
        )
        events = self._attach_event_distances(
            events=events,
            center_lat=latitude,
            center_lon=longitude,
        )

        holiday_years = {now_utc.year, (now_utc + self._days_delta(self.config.lookahead_days)).year}
        holidays: list[PublicHoliday] = []
        holiday_meta: list[dict[str, Any]] = []
        for year in sorted(holiday_years):
            items, meta = self.providers.fetch_us_public_holidays(year=year)
            holidays.extend(items)
            holiday_meta.append(meta)
        holidays = [h for h in holidays if within_days(now_utc, h.day, self.config.lookahead_days)]

        steps.append(
            StepLog(
                module="market_watch_agent.signal_collection",
                prompt={
                    "lookahead_days": self.config.lookahead_days,
                    "event_radius_km": self.config.event_radius_km,
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
        )

        weather_alerts, weather_step = self._analyze_weather(weather=weather, base_context=base_context)
        event_alerts, event_step, scored_events = self._analyze_events(events=events, base_context=base_context)
        demand_alerts, demand_step = self._analyze_demand(
            holidays=holidays,
            scored_events=scored_events,
            base_context=base_context,
        )
        steps.extend([weather_step, event_step, demand_step])

        candidates = weather_alerts + event_alerts + demand_alerts
        selected = self._select_alerts(candidates)
        steps.append(
            StepLog(
                module="market_watch_agent.alert_decision",
                prompt={"max_alerts_per_run": self.config.max_alerts_per_run},
                response={
                    "candidate_count": len(candidates),
                    "selected_count": len(selected),
                    "selected_titles": [c.title for c in selected[:6]],
                },
            )
        )

        # Insufficient evidence guard: all providers yielded empty data + no selected alerts.
        insufficient_data = not weather and not events and not holidays and not selected
        if insufficient_data:
            steps.append(
                StepLog(
                    module="market_watch_agent.inbox_write",
                    prompt={"persist_alerts": persist_alerts, "attempted": 0},
                    response={"status": "skipped", "reason": "insufficient_data"},
                )
            )
            steps.append(
                StepLog(
                    module="market_watch_agent.answer_generation",
                    prompt={"mode": "insufficient_data"},
                    response={"text": NO_EVIDENCE_RESPONSE},
                )
            )
            return MarketWatchRunOutcome(response=NO_EVIDENCE_RESPONSE, steps=steps, alerts=[], inserted_count=0)

        records = self._to_records(base_context=base_context, candidates=selected)
        inserted_count = 0
        if persist_alerts and records:
            inserted_count = self.alert_store.insert_alerts(records)
            steps.append(
                StepLog(
                    module="market_watch_agent.inbox_write",
                    prompt={"persist_alerts": True, "attempted": len(records)},
                    response={"status": "ok", "inserted": inserted_count},
                )
            )
        elif persist_alerts:
            steps.append(
                StepLog(
                    module="market_watch_agent.inbox_write",
                    prompt={"persist_alerts": True, "attempted": 0},
                    response={"status": "skipped", "reason": "no_alerts_to_persist"},
                )
            )
        else:
            steps.append(
                StepLog(
                    module="market_watch_agent.inbox_write",
                    prompt={"persist_alerts": False, "attempted": len(records)},
                    response={"status": "skipped", "reason": "on_demand_mode"},
                )
            )

        answer = self._build_answer(
            alerts=records,
            lookahead_days=self.config.lookahead_days,
        )
        steps.append(
            StepLog(
                module="market_watch_agent.answer_generation",
                prompt={"mode": "deterministic_template", "alert_count": len(records)},
                response={"text": answer},
            )
        )
        return MarketWatchRunOutcome(response=answer, steps=steps, alerts=records, inserted_count=inserted_count)

    def _extract_base_context(self, context: dict[str, object]) -> dict[str, str | None]:
        """Collect shared owner/property fields used in alerts and response context."""

        return {
            "owner_id": self._context_str(context, "owner_id"),
            "owner_name": self._context_str(context, "owner_name"),
            "property_id": self._context_str(context, "property_id"),
            "property_name": self._context_str(context, "property_name"),
            "city": self._context_str(context, "city"),
            "region": self._context_str(context, "region"),
        }

    def _analyze_weather(
        self,
        *,
        weather: list[WeatherForecastDay],
        base_context: dict[str, str | None],
    ) -> tuple[list[AlertCandidate], StepLog]:
        """Create weather-driven operational alerts from severe forecast markers."""

        candidates: list[AlertCandidate] = []
        for day in weather:
            if day.wind_kph_max is not None and day.wind_kph_max >= self.config.storm_wind_kph_threshold:
                candidates.append(
                    AlertCandidate(
                        alert_type="weather",
                        severity="high",
                        title=f"Storm-level wind risk on {day.day.isoformat()}",
                        summary=(
                            f"Forecast max wind is {day.wind_kph_max:.1f} kph. "
                            "Prepare proactive guest communication and check external fixtures."
                        ),
                        start_at_utc=self._start_of_day_utc(day.day),
                        end_at_utc=self._end_of_day_utc(day.day),
                        source_name="Open-Meteo",
                        source_url="https://open-meteo.com/",
                        evidence={
                            "wind_kph_max": day.wind_kph_max,
                            "precipitation_mm": day.precipitation_mm,
                            "snowfall_cm": day.snowfall_cm,
                            "property_id": base_context.get("property_id"),
                        },
                    )
                )
            if day.precipitation_mm is not None and day.precipitation_mm >= self.config.heavy_rain_mm_threshold:
                candidates.append(
                    AlertCandidate(
                        alert_type="operations",
                        severity="medium",
                        title=f"Heavy rain expected on {day.day.isoformat()}",
                        summary=(
                            f"Forecast rain is {day.precipitation_mm:.1f} mm. "
                            "Review check-in logistics, parking guidance, and maintenance readiness."
                        ),
                        start_at_utc=self._start_of_day_utc(day.day),
                        end_at_utc=self._end_of_day_utc(day.day),
                        source_name="Open-Meteo",
                        source_url="https://open-meteo.com/",
                        evidence={
                            "wind_kph_max": day.wind_kph_max,
                            "precipitation_mm": day.precipitation_mm,
                            "snowfall_cm": day.snowfall_cm,
                            "property_id": base_context.get("property_id"),
                        },
                    )
                )
            if day.snowfall_cm is not None and day.snowfall_cm >= self.config.snow_cm_threshold:
                candidates.append(
                    AlertCandidate(
                        alert_type="operations",
                        severity="high",
                        title=f"Snow disruption risk on {day.day.isoformat()}",
                        summary=(
                            f"Forecast snowfall is {day.snowfall_cm:.1f} cm. "
                            "Plan access instructions and early guest messaging for travel disruptions."
                        ),
                        start_at_utc=self._start_of_day_utc(day.day),
                        end_at_utc=self._end_of_day_utc(day.day),
                        source_name="Open-Meteo",
                        source_url="https://open-meteo.com/",
                        evidence={
                            "wind_kph_max": day.wind_kph_max,
                            "precipitation_mm": day.precipitation_mm,
                            "snowfall_cm": day.snowfall_cm,
                            "property_id": base_context.get("property_id"),
                        },
                    )
                )

        step = StepLog(
            module="market_watch_agent.weather_analysis",
            prompt={
                "storm_wind_kph_threshold": self.config.storm_wind_kph_threshold,
                "heavy_rain_mm_threshold": self.config.heavy_rain_mm_threshold,
                "snow_cm_threshold": self.config.snow_cm_threshold,
            },
            response={
                "forecast_days": len(weather),
                "candidate_count": len(candidates),
            },
        )
        return candidates, step

    def _analyze_events(
        self,
        *,
        events: list[NearbyEvent],
        base_context: dict[str, str | None],
    ) -> tuple[list[AlertCandidate], StepLog, list[dict[str, Any]]]:
        """Create event-impact alerts using category and proximity scoring."""

        candidates: list[AlertCandidate] = []
        scored_events: list[dict[str, Any]] = []
        now = utc_now()
        for event in events:
            if event.start_at_utc is None or event.distance_km is None:
                continue
            if event.distance_km > float(self.config.event_radius_km):
                continue

            days_until = max(0, (event.start_at_utc.date() - now.date()).days)
            category_weight = self._category_weight(event.category)
            proximity_bonus = 1 if event.distance_km <= 5 else 0
            urgency_bonus = 1 if days_until <= 3 else 0
            popularity_bonus = 1 if (event.popularity_hint or "").lower() == "high" else 0
            impact_score = category_weight + proximity_bonus + urgency_bonus + popularity_bonus
            severity = "high" if impact_score >= 4 else "medium" if impact_score >= 3 else "low"
            scored_events.append(
                {
                    "name": event.name,
                    "distance_km": round(event.distance_km, 2),
                    "days_until": days_until,
                    "category": event.category,
                    "impact_score": impact_score,
                    "severity": severity,
                }
            )
            if severity == "low":
                continue

            candidates.append(
                AlertCandidate(
                    alert_type="event",
                    severity=severity,
                    title=f"Nearby {severity}-impact event: {event.name}",
                    summary=(
                        f"{event.name} is about {event.distance_km:.1f} km away in {days_until} day(s). "
                        "Consider price and staffing adjustments for possible demand lift."
                    ),
                    start_at_utc=event.start_at_utc.astimezone(UTC).isoformat(),
                    end_at_utc=None,
                    source_name="Ticketmaster",
                    source_url=event.source_url,
                    evidence={
                        "event_name": event.name,
                        "category": event.category,
                        "distance_km": event.distance_km,
                        "impact_score": impact_score,
                        "property_id": base_context.get("property_id"),
                    },
                )
            )

        step = StepLog(
            module="market_watch_agent.event_analysis",
            prompt={"event_radius_km": self.config.event_radius_km},
            response={
                "raw_events_count": len(events),
                "scored_events_count": len(scored_events),
                "candidate_count": len(candidates),
            },
        )
        return candidates, step, scored_events

    def _analyze_demand(
        self,
        *,
        holidays: list[PublicHoliday],
        scored_events: list[dict[str, Any]],
        base_context: dict[str, str | None],
    ) -> tuple[list[AlertCandidate], StepLog]:
        """Generate demand alerts from holiday windows plus concentrated nearby events."""

        candidates: list[AlertCandidate] = []
        now = utc_now()
        upcoming_holidays = sorted(
            [h for h in holidays if within_days(now, h.day, self.config.lookahead_days)],
            key=lambda h: h.day,
        )
        medium_plus_events = [e for e in scored_events if str(e.get("severity")) in {"medium", "high"}]

        if upcoming_holidays or len(medium_plus_events) >= 2:
            holiday_label = upcoming_holidays[0].name if upcoming_holidays else "upcoming period"
            severity = "high" if (upcoming_holidays and len(medium_plus_events) >= 2) else "medium"
            candidates.append(
                AlertCandidate(
                    alert_type="demand",
                    severity=severity,
                    title=f"Demand opportunity signal around {holiday_label}",
                    summary=(
                        f"Detected {len(medium_plus_events)} medium/high nearby events "
                        f"and {len(upcoming_holidays)} holiday signal(s) in the next {self.config.lookahead_days} days. "
                        "Consider testing higher nightly rates and minimum-stay strategy."
                    ),
                    start_at_utc=self._start_of_day_utc(upcoming_holidays[0].day) if upcoming_holidays else None,
                    end_at_utc=None,
                    source_name="Nager.Date + Ticketmaster",
                    source_url="https://date.nager.at/",
                    evidence={
                        "holiday_count": len(upcoming_holidays),
                        "event_signal_count": len(medium_plus_events),
                        "property_id": base_context.get("property_id"),
                    },
                )
            )

        step = StepLog(
            module="market_watch_agent.demand_analysis",
            prompt={"lookahead_days": self.config.lookahead_days},
            response={
                "holiday_count": len(upcoming_holidays),
                "event_signal_count": len(medium_plus_events),
                "candidate_count": len(candidates),
            },
        )
        return candidates, step

    def _select_alerts(self, candidates: list[AlertCandidate]) -> list[AlertCandidate]:
        """Sort candidates by severity and start time, then cap by configured max size."""

        severity_rank = {"high": 0, "medium": 1, "low": 2}
        ordered = sorted(
            candidates,
            key=lambda c: (
                severity_rank.get(c.severity, 9),
                c.start_at_utc or "9999-12-31T23:59:59+00:00",
            ),
        )
        return ordered[: self.config.max_alerts_per_run]

    def _to_records(
        self,
        *,
        base_context: dict[str, str | None],
        candidates: list[AlertCandidate],
    ) -> list[MarketAlertRecord]:
        """Convert candidates to persistent record format with deterministic IDs."""

        created_at = utc_now_iso()
        records: list[MarketAlertRecord] = []
        for candidate in candidates:
            records.append(
                MarketAlertRecord(
                    id=build_alert_id(
                        owner_id=base_context.get("owner_id"),
                        property_id=base_context.get("property_id"),
                        alert_type=candidate.alert_type,
                        title=candidate.title,
                        start_at_utc=candidate.start_at_utc,
                    ),
                    created_at_utc=created_at,
                    owner_id=base_context.get("owner_id"),
                    property_id=base_context.get("property_id"),
                    property_name=base_context.get("property_name"),
                    city=base_context.get("city"),
                    region=base_context.get("region"),
                    alert_type=candidate.alert_type,
                    severity=candidate.severity,
                    title=candidate.title,
                    summary=candidate.summary,
                    start_at_utc=candidate.start_at_utc,
                    end_at_utc=candidate.end_at_utc,
                    source_name=candidate.source_name,
                    source_url=candidate.source_url,
                    evidence=candidate.evidence,
                )
            )
        return records

    def _build_answer(
        self,
        *,
        alerts: list[MarketAlertRecord],
        lookahead_days: int,
    ) -> str:
        """Build concise evidence-grounded answer text without speculative language."""

        if not alerts:
            return (
                f"No notable market-watch signals were detected for the next {lookahead_days} days "
                "based on current event, weather, and holiday data."
            )

        lines: list[str] = [
            f"Detected {len(alerts)} actionable market signal(s) for the next {lookahead_days} days:",
        ]
        citations: list[str] = []
        for idx, alert in enumerate(alerts[:5], start=1):
            when = f" (starts {alert.start_at_utc})" if alert.start_at_utc else ""
            lines.append(f"{idx}. [{alert.severity}] {alert.title}{when}: {alert.summary}")
            citation = f"{alert.source_name}{' + ' + alert.source_url if alert.source_url else ''}"
            if citation not in citations:
                citations.append(citation)

        if citations:
            lines.append("")
            lines.append("Sources:")
            lines.extend([f"- {citation}" for citation in citations[:5]])

        text = "\n".join(lines)
        return self._cap_words(text, self.config.max_answer_words)

    def _attach_event_distances(
        self,
        *,
        events: list[NearbyEvent],
        center_lat: float,
        center_lon: float,
    ) -> list[NearbyEvent]:
        """Attach haversine distance for each event when venue coordinates are present."""

        out: list[NearbyEvent] = []
        for event in events:
            if event.latitude is not None and event.longitude is not None:
                event.distance_km = self._haversine_km(center_lat, center_lon, event.latitude, event.longitude)
            out.append(event)
        return out

    def _context_str(self, context: dict[str, object], key: str) -> str | None:
        """Read optional context string safely."""

        value = context.get(key)
        if isinstance(value, str):
            trimmed = value.strip()
            return trimmed or None
        return None

    def _context_float(self, context: dict[str, object], *keys: str) -> float | None:
        """Read optional numeric values from context using multiple key aliases."""

        for key in keys:
            value = context.get(key)
            if value is None:
                continue
            # Parse only scalar numeric-like values; ignore dict/list/object payloads safely.
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
            # Non-scalar objects are intentionally ignored.
            else:
                continue
        return None

    def _haversine_km(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Great-circle distance in kilometers."""

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

    def _category_weight(self, category: str | None) -> int:
        """Map event categories to coarse impact priors."""

        label = (category or "").strip().lower()
        if any(token in label for token in ["music", "concert", "sports", "festival"]):
            return 2
        if any(token in label for token in ["theatre", "family", "film", "conference"]):
            return 1
        return 0

    def _start_of_day_utc(self, day_value: date) -> str:
        """Build ISO UTC timestamp at day start."""

        return datetime(day_value.year, day_value.month, day_value.day, tzinfo=UTC).isoformat()

    def _end_of_day_utc(self, day_value: date) -> str:
        """Build ISO UTC timestamp at day end."""

        return datetime(day_value.year, day_value.month, day_value.day, 23, 59, 59, tzinfo=UTC).isoformat()

    def _days_delta(self, days: int) -> timedelta:
        """Helper returning timedelta without importing in many call sites."""

        return timedelta(days=max(0, int(days)))

    def _cap_words(self, text: str, max_words: int) -> str:
        """Cap answer text length to keep output cost predictable."""

        words = text.split()
        if len(words) <= max_words:
            return text
        return " ".join(words[:max_words]) + " ..."
