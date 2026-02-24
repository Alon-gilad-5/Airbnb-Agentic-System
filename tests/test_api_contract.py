"""Phase 0 contract tests: verify agent response shapes, module names, and error paths.

These tests run with dummy services (no live API calls) and assert the exact
response contract that the FastAPI endpoints depend on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from app.agents.reviews_agent import ReviewsAgent, ReviewsAgentConfig
from app.agents.market_watch_agent import MarketWatchAgent, MarketWatchAgentConfig
from app.agents.router_agent import RouterAgent
from app.services.pinecone_retriever import RetrievedReview
from app.services.web_review_ingest import WebIngestResult
from app.services.web_review_scraper import ScrapedReview


# -- Dummy services --


@dataclass
class DummyEmbeddingService:
    is_available: bool = True

    def embed_query(self, text: str) -> list[float]:
        return [0.1, 0.2]


@dataclass
class DummyRetriever:
    is_available: bool = True
    _matches: list[RetrievedReview] | None = None

    def query(
        self,
        *,
        embedding: list[float],
        top_k: int,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[RetrievedReview]:
        if self._matches is not None:
            return self._matches
        return []


@dataclass
class DummyChatService:
    is_available: bool = True
    model: str = "test-model"
    _response: str = "Test LLM answer about wifi."

    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        if not self.is_available:
            raise RuntimeError("Chat service unavailable")
        return self._response


class DummyWebScraper:
    def __init__(self, *, available: bool = False) -> None:
        self.is_available = available

    def scrape_reviews(self, **kwargs: Any) -> tuple[list[ScrapedReview], dict[str, Any]]:
        return [], {"status": "disabled", "raw_count": 0, "deduped_count": 0}


class DummyWebIngest:
    is_available = False

    def upsert_scraped_reviews(self, **kwargs: Any) -> WebIngestResult:
        return WebIngestResult(attempted=0, upserted=0, vector_ids=[], namespace="test")


# -- Market watch dummies --


class DummyProviders:
    def fetch_weather_forecast(self, **kwargs: Any) -> tuple[list, dict]:
        return [], {"source": "test", "status": "ok"}

    def fetch_ticketmaster_events(self, **kwargs: Any) -> tuple[list, dict]:
        return [], {"source": "test", "status": "ok"}

    def fetch_us_public_holidays(self, **kwargs: Any) -> tuple[list, dict]:
        return [], {"source": "test", "status": "ok"}


class DummyAlertStore:
    def insert_alerts(self, records: list) -> int:
        return len(records)


# -- Reviews Agent contract tests --


REVIEWS_MODULE_NAMES = {
    "reviews_agent.retrieval",
    "reviews_agent.web_scrape",
    "reviews_agent.web_quarantine_upsert",
    "reviews_agent.evidence_guard",
    "reviews_agent.answer_generation",
    "reviews_agent.hallucination_guard",
}


def _make_reviews_agent(
    *,
    embedding_available: bool = True,
    retriever_available: bool = True,
    retriever_matches: list[RetrievedReview] | None = None,
    chat_available: bool = True,
    chat_response: str = "Test wifi answer.",
) -> ReviewsAgent:
    return ReviewsAgent(
        embedding_service=DummyEmbeddingService(is_available=embedding_available),
        retriever=DummyRetriever(is_available=retriever_available, _matches=retriever_matches),
        chat_service=DummyChatService(is_available=chat_available, _response=chat_response),
        web_scraper=DummyWebScraper(available=False),
        web_ingest_service=DummyWebIngest(),
        config=ReviewsAgentConfig(relevance_score_threshold=0.40),
    )


def test_reviews_no_matches_returns_no_evidence() -> None:
    agent = _make_reviews_agent(retriever_matches=[])
    result = agent.run("What do guests say about wifi?", context={"property_id": "42409434", "region": "los angeles"})
    assert result.response == "I couldn't find enough data to answer your question."
    module_names = {s.module for s in result.steps}
    assert "reviews_agent.retrieval" in module_names
    assert "reviews_agent.evidence_guard" in module_names


def test_reviews_with_matches_produces_answer_and_all_modules() -> None:
    matches = [
        RetrievedReview(vector_id="v1", score=0.85, metadata={
            "review_text": "Wifi was excellent and fast.",
            "property_id": "42409434", "region": "los angeles", "review_date": "2024-01-01",
        }),
        RetrievedReview(vector_id="v2", score=0.72, metadata={
            "review_text": "Good internet connection.",
            "property_id": "42409434", "region": "los angeles", "review_date": "2024-02-15",
        }),
        RetrievedReview(vector_id="v3", score=0.68, metadata={
            "review_text": "Wifi worked well throughout stay.",
            "property_id": "42409434", "region": "los angeles", "review_date": "2024-03-10",
        }),
    ]
    agent = _make_reviews_agent(retriever_matches=matches)
    result = agent.run("What do guests say about wifi?", context={"property_id": "42409434", "region": "los angeles"})

    assert result.response is not None
    assert isinstance(result.response, str)
    assert len(result.response) > 0

    module_names = [s.module for s in result.steps]
    assert "reviews_agent.retrieval" in module_names
    assert "reviews_agent.evidence_guard" in module_names
    assert "reviews_agent.answer_generation" in module_names
    assert "reviews_agent.hallucination_guard" in module_names

    for step in result.steps:
        assert step.module in REVIEWS_MODULE_NAMES
        assert isinstance(step.prompt, dict)
        assert isinstance(step.response, dict)


def test_reviews_embedding_unavailable() -> None:
    agent = _make_reviews_agent(embedding_available=False)
    result = agent.run("What do guests say about wifi?")
    assert "Embedding service is not configured" in result.response
    assert len(result.steps) == 0


def test_reviews_retriever_unavailable() -> None:
    agent = _make_reviews_agent(retriever_available=False)
    result = agent.run("What do guests say about wifi?")
    assert "Pinecone is not configured" in result.response
    assert len(result.steps) == 0


def test_reviews_chat_unavailable_uses_deterministic_fallback() -> None:
    matches = [
        RetrievedReview(vector_id="v1", score=0.85, metadata={
            "review_text": "Wifi was excellent.", "property_id": "p1", "region": "la",
        }),
        RetrievedReview(vector_id="v2", score=0.72, metadata={
            "review_text": "Good internet.", "property_id": "p1", "region": "la",
        }),
        RetrievedReview(vector_id="v3", score=0.68, metadata={
            "review_text": "Wifi worked well.", "property_id": "p1", "region": "la",
        }),
    ]
    agent = _make_reviews_agent(retriever_matches=matches, chat_available=False)
    result = agent.run("What do guests say about wifi?")
    assert "LLM synthesis service is currently unavailable" in result.response
    module_names = {s.module for s in result.steps}
    assert "reviews_agent.retrieval" in module_names
    assert "reviews_agent.evidence_guard" in module_names
    assert "reviews_agent.hallucination_guard" in module_names


def test_reviews_thin_evidence_produces_disclaimer() -> None:
    matches = [
        RetrievedReview(vector_id="v1", score=0.85, metadata={
            "review_text": "Wifi was excellent.", "property_id": "p1", "region": "la",
        }),
    ]
    agent = _make_reviews_agent(retriever_matches=matches)
    result = agent.run("What do guests say about wifi?")
    assert "Evidence is limited" in result.response


# -- Market Watch Agent contract tests --


MARKET_WATCH_MODULE_NAMES = {
    "market_watch_agent.signal_collection",
    "market_watch_agent.weather_analysis",
    "market_watch_agent.event_analysis",
    "market_watch_agent.demand_analysis",
    "market_watch_agent.alert_decision",
    "market_watch_agent.inbox_write",
    "market_watch_agent.answer_generation",
}


def _make_market_agent() -> MarketWatchAgent:
    return MarketWatchAgent(
        providers=DummyProviders(),
        alert_store=DummyAlertStore(),
        config=MarketWatchAgentConfig(),
    )


def test_market_watch_missing_coordinates() -> None:
    agent = _make_market_agent()
    result = agent.run("Any nearby events?", context={})
    assert result.response == "I couldn't find enough data to answer your question."
    module_names = {s.module for s in result.steps}
    assert "market_watch_agent.signal_collection" in module_names
    assert "market_watch_agent.answer_generation" in module_names


def test_market_watch_with_coordinates_returns_full_trace() -> None:
    agent = _make_market_agent()
    result = agent.run("Any nearby events?", context={"latitude": 34.0522, "longitude": -118.2437})
    assert isinstance(result.response, str)
    module_names = [s.module for s in result.steps]
    for step in result.steps:
        assert step.module in MARKET_WATCH_MODULE_NAMES
        assert isinstance(step.prompt, dict)
        assert isinstance(step.response, dict)


def test_market_watch_autonomous_returns_outcome() -> None:
    agent = _make_market_agent()
    outcome = agent.run_autonomous(context={"latitude": 34.0522, "longitude": -118.2437})
    assert isinstance(outcome.response, str)
    assert isinstance(outcome.steps, list)
    assert isinstance(outcome.alerts, list)
    assert isinstance(outcome.inserted_count, int)


# -- Router contract tests --


def test_router_reviews_keywords() -> None:
    router = RouterAgent()
    decision, step = router.route("What do guests say about wifi?")
    assert decision.agent_name == "reviews_agent"
    assert step.module == "router_agent"
    assert step.response["selected_agent"] == "reviews_agent"


def test_router_market_keywords() -> None:
    router = RouterAgent()
    decision, step = router.route("Any nearby events next week?")
    assert decision.agent_name == "market_watch_agent"
    assert step.module == "router_agent"


def test_router_fallback_to_reviews() -> None:
    router = RouterAgent()
    decision, step = router.route("Tell me something interesting")
    assert decision.agent_name == "reviews_agent"
    assert "Fallback" in decision.reason


def test_router_graph_produces_same_result() -> None:
    router = RouterAgent()
    decision_direct, step_direct = router.route("What about wifi?")
    decision_graph, step_graph = router.route_via_graph("What about wifi?")
    assert decision_direct.agent_name == decision_graph.agent_name
    assert step_direct.module == step_graph.module
