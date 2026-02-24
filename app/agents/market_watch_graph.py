"""Compatibility wrapper for the market-watch pipeline.

Delegates to the LangChain-first MarketWatchPipeline defined in
market_watch_agent while preserving the original build/invoke interface
for existing callers and tests.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from app.agents.market_watch_agent import (
    AlertCandidate,
    MarketWatchPipeline,
    NO_EVIDENCE_RESPONSE,
)
from app.schemas import StepLog
from app.services.market_alert_store import MarketAlertRecord
from app.services.market_data_providers import (
    NearbyEvent,
    PublicHoliday,
    WeatherForecastDay,
)

__all__ = [
    "NO_EVIDENCE_RESPONSE",
    "AlertCandidate",
    "MarketWatchState",
    "build_market_watch_graph",
]


class MarketWatchState(TypedDict, total=False):
    """Pipeline state schema kept for type-annotation compatibility."""

    prompt: str
    context: dict[str, Any]
    persist_alerts: bool
    base_context: dict[str, str | None]
    latitude: float | None
    longitude: float | None
    has_coordinates: bool
    weather: list[WeatherForecastDay]
    events: list[NearbyEvent]
    holidays: list[PublicHoliday]
    weather_alerts: list[AlertCandidate]
    event_alerts: list[AlertCandidate]
    scored_events: list[dict[str, Any]]
    demand_alerts: list[AlertCandidate]
    selected: list[AlertCandidate]
    records: list[MarketAlertRecord]
    inserted_count: int
    answer: str
    steps: Annotated[list[StepLog], operator.add]


def build_market_watch_graph(
    *,
    providers: Any,
    alert_store: Any,
    config: Any,
) -> MarketWatchPipeline:
    """Build the market-watch pipeline with injected services.

    Returns a MarketWatchPipeline whose ``.invoke()`` method accepts and
    returns the same dict shape as the original StateGraph for full backward
    compatibility.
    """

    return MarketWatchPipeline(
        providers=providers,
        alert_store=alert_store,
        config=config,
    )
