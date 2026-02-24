"""Node-level unit tests for the market-watch LangGraph StateGraph."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, date
from typing import Any

import pytest

from app.agents.market_watch_graph import build_market_watch_graph, NO_EVIDENCE_RESPONSE
from app.agents.market_watch_agent import MarketWatchAgentConfig
from app.services.market_data_providers import NearbyEvent, PublicHoliday, WeatherForecastDay


# -- Dummy services --


class _DummyProviders:
    """Configurable dummy that returns injected signal data."""

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

    def fetch_weather_forecast(self, **kwargs: Any) -> tuple[list[WeatherForecastDay], dict]:
        return self._weather, {"source": "test", "status": "ok"}

    def fetch_ticketmaster_events(self, **kwargs: Any) -> tuple[list[NearbyEvent], dict]:
        return self._events, {"source": "test", "status": "ok"}

    def fetch_us_public_holidays(self, **kwargs: Any) -> tuple[list[PublicHoliday], dict]:
        return self._holidays, {"source": "test", "status": "ok"}


class _DummyAlertStore:
    def __init__(self) -> None:
        self.inserted: list[Any] = []

    def insert_alerts(self, records: list) -> int:
        self.inserted.extend(records)
        return len(records)


def _build(
    *,
    weather: list[WeatherForecastDay] | None = None,
    events: list[NearbyEvent] | None = None,
    holidays: list[PublicHoliday] | None = None,
) -> tuple[Any, _DummyAlertStore]:
    store = _DummyAlertStore()
    cfg = MarketWatchAgentConfig()
    graph = build_market_watch_graph(
        providers=_DummyProviders(weather=weather, events=events, holidays=holidays),
        alert_store=store,
        config=cfg,
    )
    return graph, store


def test_missing_coordinates_early_exit() -> None:
    graph, _ = _build()
    result = graph.invoke({"prompt": "events?", "context": {}, "persist_alerts": False, "steps": []})
    assert result["answer"] == NO_EVIDENCE_RESPONSE
    module_names = {s.module for s in result["steps"]}
    assert "market_watch_agent.signal_collection" in module_names
    assert "market_watch_agent.answer_generation" in module_names


def test_empty_signals_returns_no_evidence() -> None:
    graph, _ = _build()
    result = graph.invoke({
        "prompt": "events?",
        "context": {"latitude": 34.05, "longitude": -118.24},
        "persist_alerts": False,
        "steps": [],
    })
    module_names = {s.module for s in result["steps"]}
    assert "market_watch_agent.signal_collection" in module_names
    assert "market_watch_agent.inbox_write" in module_names


def test_weather_alert_generated_for_high_wind() -> None:
    tomorrow = date.today() + timedelta(days=1)
    weather = [
        WeatherForecastDay(
            day=tomorrow,
            weather_code=0,
            temp_max_c=25.0,
            temp_min_c=15.0,
            wind_kph_max=60.0,
            precipitation_mm=5.0,
            snowfall_cm=0.0,
        ),
    ]
    graph, _ = _build(weather=weather)
    result = graph.invoke({
        "prompt": "weather?",
        "context": {"latitude": 34.05, "longitude": -118.24},
        "persist_alerts": False,
        "steps": [],
    })
    weather_steps = [s for s in result["steps"] if s.module == "market_watch_agent.weather_analysis"]
    assert len(weather_steps) == 1
    assert weather_steps[0].response["candidate_count"] >= 1

    answer_steps = [s for s in result["steps"] if s.module == "market_watch_agent.answer_generation"]
    assert len(answer_steps) == 1
    assert "Storm-level wind" in result.get("answer", "")


def test_persist_alerts_inserts_into_store() -> None:
    tomorrow = date.today() + timedelta(days=1)
    weather = [
        WeatherForecastDay(day=tomorrow, weather_code=0, temp_max_c=25.0, temp_min_c=15.0, wind_kph_max=60.0, precipitation_mm=5.0, snowfall_cm=0.0),
    ]
    graph, store = _build(weather=weather)
    result = graph.invoke({
        "prompt": "autonomous market watch cycle",
        "context": {"latitude": 34.05, "longitude": -118.24, "property_id": "p-1"},
        "persist_alerts": True,
        "steps": [],
    })
    assert result.get("inserted_count", 0) > 0
    assert len(store.inserted) > 0

    inbox_steps = [s for s in result["steps"] if s.module == "market_watch_agent.inbox_write"]
    assert len(inbox_steps) == 1
    assert inbox_steps[0].response["status"] == "ok"


def test_on_demand_mode_skips_persistence() -> None:
    tomorrow = date.today() + timedelta(days=1)
    weather = [
        WeatherForecastDay(day=tomorrow, weather_code=0, temp_max_c=25.0, temp_min_c=15.0, wind_kph_max=60.0, precipitation_mm=5.0, snowfall_cm=0.0),
    ]
    graph, store = _build(weather=weather)
    result = graph.invoke({
        "prompt": "weather?",
        "context": {"latitude": 34.05, "longitude": -118.24},
        "persist_alerts": False,
        "steps": [],
    })
    assert len(store.inserted) == 0
    inbox_steps = [s for s in result["steps"] if s.module == "market_watch_agent.inbox_write"]
    assert len(inbox_steps) == 1
    assert inbox_steps[0].response["reason"] == "on_demand_mode"


def test_all_module_names_are_valid() -> None:
    """Every step module produced by the graph must be in the known set."""
    valid_modules = {
        "market_watch_agent.signal_collection",
        "market_watch_agent.weather_analysis",
        "market_watch_agent.event_analysis",
        "market_watch_agent.demand_analysis",
        "market_watch_agent.alert_decision",
        "market_watch_agent.inbox_write",
        "market_watch_agent.answer_generation",
    }
    tomorrow = date.today() + timedelta(days=1)
    weather = [WeatherForecastDay(day=tomorrow, weather_code=0, temp_max_c=25.0, temp_min_c=15.0, wind_kph_max=60.0, precipitation_mm=5.0, snowfall_cm=0.0)]
    graph, _ = _build(weather=weather)
    result = graph.invoke({
        "prompt": "weather?",
        "context": {"latitude": 34.05, "longitude": -118.24},
        "persist_alerts": False,
        "steps": [],
    })
    for step in result["steps"]:
        assert step.module in valid_modules, f"Unknown module: {step.module}"
