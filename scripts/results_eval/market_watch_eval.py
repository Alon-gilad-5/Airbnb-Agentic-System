"""Offline evaluator for market-watch alert detection and determinism."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
import time
from typing import Any

from app.agents.market_watch_agent import MarketWatchAgent, MarketWatchAgentConfig, NO_EVIDENCE_RESPONSE
from app.schemas import StepLog
from app.services.market_data_providers import NearbyEvent, PublicHoliday, WeatherForecastDay
from scripts.results_eval.common import compute_reliability, compute_task_success_rate, parse_cases


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


class _DummyAlertStore:
    def __init__(self) -> None:
        self.records: dict[str, Any] = {}

    def insert_alerts(self, records: list[Any]) -> int:
        for record in records:
            self.records[record.id] = record
        return len(records)


VALID_MODULES = {
    "market_watch_agent.signal_collection",
    "market_watch_agent.weather_analysis",
    "market_watch_agent.event_analysis",
    "market_watch_agent.demand_analysis",
    "market_watch_agent.alert_decision",
    "market_watch_agent.inbox_write",
    "market_watch_agent.answer_generation",
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


def evaluate_market_watch(
    *,
    repo_root: Path,
    split: str,
) -> dict[str, Any]:
    """Evaluate market-watch alert detection rules on synthetic fixtures."""

    cases = parse_cases(repo_root / "eval" / "cases" / "market_watch_cases.jsonl", split=split)
    case_results: list[dict[str, Any]] = []
    alert_detection_hits = 0
    alert_detection_total = 0

    for case in cases:
        started = time.perf_counter()
        exception = None
        contract_ok = False
        trace_ok = False
        passed = False
        failure_reason = "uninitialized"
        try:
            fixtures = case.raw.get("fixtures") if isinstance(case.raw.get("fixtures"), dict) else {}
            store = _DummyAlertStore()
            agent = MarketWatchAgent(
                providers=_DummyProviders(
                    weather=_to_weather(fixtures.get("weather", [])),
                    events=_to_events(fixtures.get("events", [])),
                    holidays=_to_holidays(fixtures.get("holidays", [])),
                ),
                alert_store=store,
                config=MarketWatchAgentConfig(),
            )
            outcome = agent.run_autonomous(case.context)
            modules = [step.module for step in outcome.steps]
            contract_ok = (
                isinstance(outcome.response, str)
                and isinstance(outcome.inserted_count, int)
                and all(_is_step_log(step) for step in outcome.steps)
            )
            required_modules = set(case.expected.get("required_modules", ["market_watch_agent.answer_generation"]))
            trace_ok = set(modules).issubset(VALID_MODULES) and required_modules.issubset(set(modules))

            if bool(case.expected.get("no_evidence")):
                passed = outcome.response.startswith(NO_EVIDENCE_RESPONSE)
                failure_reason = None if passed else "expected_no_evidence_response"
            else:
                alert_detection_total += 1
                produced_types = {record.alert_type for record in outcome.alerts}
                produced_severities = {record.severity for record in outcome.alerts}
                expected_types = set(case.expected.get("expected_alert_types", []))
                expected_severities = set(case.expected.get("expected_severities", []))
                min_alert_count = int(case.expected.get("min_alert_count", 1))
                type_ok = expected_types.issubset(produced_types)
                severity_ok = (not expected_severities) or expected_severities.issubset(produced_severities)
                count_ok = len(outcome.alerts) >= min_alert_count
                rerun_ok = True
                if bool(case.expected.get("dedupe_stable")):
                    second = agent.run_autonomous(case.context)
                    first_ids = sorted(record.id for record in outcome.alerts)
                    second_ids = sorted(record.id for record in second.alerts)
                    rerun_ok = first_ids == second_ids
                detected = type_ok and severity_ok and count_ok and rerun_ok
                alert_detection_hits += 1 if detected else 0
                passed = detected
                failure_reason = None if passed else "alert_expectation_mismatch"
        except Exception as exc:  # pragma: no cover - defensive guard
            exception = f"{type(exc).__name__}: {exc}"
            failure_reason = "exception"

        latency_ms = (time.perf_counter() - started) * 1000.0
        case_results.append(
            {
                "agent": "market_watch",
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
    alert_detection_accuracy = (alert_detection_hits / alert_detection_total) if alert_detection_total else 0.0
    task_success_rate = compute_task_success_rate(case_results)
    return {
        "agent": "market_watch",
        "split": split,
        "primary_metric_name": "alert_detection_accuracy",
        "primary_metric": round(alert_detection_accuracy, 4),
        "metrics": {
            "case_count": len(case_results),
            "alert_detection_accuracy": round(alert_detection_accuracy, 4),
            "task_success_rate": task_success_rate,
        },
        "reliability": reliability,
        "task_success_rate": task_success_rate,
        "case_results": case_results,
    }

