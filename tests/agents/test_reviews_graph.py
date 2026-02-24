"""Node-level unit tests for the reviews LangGraph StateGraph."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from app.agents.reviews_graph import build_reviews_graph, NO_EVIDENCE_RESPONSE
from app.agents.reviews_agent import ReviewsAgentConfig
from app.services.pinecone_retriever import RetrievedReview
from app.services.web_review_ingest import WebIngestResult
from app.services.web_review_scraper import ScrapedReview


# -- Dummy services --


@dataclass
class _DummyEmbedding:
    is_available: bool = True

    def embed_query(self, text: str) -> list[float]:
        return [0.1, 0.2]


@dataclass
class _DummyRetriever:
    is_available: bool = True
    _matches: list[RetrievedReview] | None = None

    def query(self, *, embedding: list[float], top_k: int, metadata_filter: Any = None) -> list[RetrievedReview]:
        return self._matches if self._matches is not None else []


@dataclass
class _DummyChat:
    is_available: bool = True
    model: str = "test-model"
    _response: str = "LLM response about wifi quality."

    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        return self._response


class _DummyScraper:
    is_available = False

    def scrape_reviews(self, **kwargs: Any) -> tuple[list[ScrapedReview], dict[str, Any]]:
        return [], {"status": "disabled", "raw_count": 0, "deduped_count": 0}


class _DummyIngest:
    is_available = False

    def upsert_scraped_reviews(self, **kwargs: Any) -> WebIngestResult:
        return WebIngestResult(attempted=0, upserted=0, vector_ids=[], namespace="test")


def _build(
    *,
    matches: list[RetrievedReview] | None = None,
    embedding_available: bool = True,
    retriever_available: bool = True,
    chat_available: bool = True,
    chat_response: str = "LLM answer.",
) -> Any:
    cfg = ReviewsAgentConfig(relevance_score_threshold=0.40)
    return build_reviews_graph(
        embedding_service=_DummyEmbedding(is_available=embedding_available),
        retriever=_DummyRetriever(is_available=retriever_available, _matches=matches),
        chat_service=_DummyChat(is_available=chat_available, _response=chat_response),
        web_scraper=_DummyScraper(),
        web_ingest_service=_DummyIngest(),
        config=cfg,
    )


def test_graph_no_matches_returns_no_evidence() -> None:
    graph = _build(matches=[])
    result = graph.invoke({"prompt": "wifi quality?", "context": {}, "steps": []})
    assert result["final_answer"] == NO_EVIDENCE_RESPONSE
    module_names = {s.module for s in result["steps"]}
    assert "reviews_agent.retrieval" in module_names
    assert "reviews_agent.evidence_guard" in module_names


def test_graph_with_matches_produces_answer_steps() -> None:
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
    graph = _build(matches=matches)
    result = graph.invoke({"prompt": "wifi quality?", "context": {}, "steps": []})
    assert result["final_answer"]
    assert len(result["final_answer"]) > 0
    module_names = [s.module for s in result["steps"]]
    assert "reviews_agent.retrieval" in module_names
    assert "reviews_agent.evidence_guard" in module_names
    assert "reviews_agent.answer_generation" in module_names
    assert "reviews_agent.hallucination_guard" in module_names


def test_graph_embedding_unavailable_early_exit() -> None:
    graph = _build(embedding_available=False)
    result = graph.invoke({"prompt": "wifi?", "context": {}, "steps": []})
    assert "Embedding service is not configured" in result["final_answer"]


def test_graph_retriever_unavailable_early_exit() -> None:
    graph = _build(retriever_available=False)
    result = graph.invoke({"prompt": "wifi?", "context": {}, "steps": []})
    assert "Pinecone is not configured" in result["final_answer"]


def test_graph_chat_unavailable_uses_deterministic_summary() -> None:
    matches = [
        RetrievedReview(vector_id="v1", score=0.85, metadata={
            "review_text": "Wifi was great.", "property_id": "p1", "region": "la",
        }),
        RetrievedReview(vector_id="v2", score=0.72, metadata={
            "review_text": "Good connection.", "property_id": "p1", "region": "la",
        }),
        RetrievedReview(vector_id="v3", score=0.65, metadata={
            "review_text": "Worked fine.", "property_id": "p1", "region": "la",
        }),
    ]
    graph = _build(matches=matches, chat_available=False)
    result = graph.invoke({"prompt": "wifi?", "context": {}, "steps": []})
    assert "strongest matching reviews" in result["final_answer"]


def test_graph_thin_evidence_adds_disclaimer() -> None:
    matches = [
        RetrievedReview(vector_id="v1", score=0.85, metadata={
            "review_text": "Wifi was ok.", "property_id": "p1", "region": "la",
        }),
    ]
    graph = _build(matches=matches)
    result = graph.invoke({"prompt": "wifi?", "context": {}, "steps": []})
    assert "Evidence is limited" in result["final_answer"]


def test_graph_metadata_filter_with_property_and_region() -> None:
    graph = _build(matches=[])
    result = graph.invoke({
        "prompt": "wifi?",
        "context": {"property_id": "p-123", "region": "  SAN FRANCISCO  "},
        "steps": [],
    })
    retrieval_steps = [s for s in result["steps"] if s.module == "reviews_agent.retrieval"]
    assert len(retrieval_steps) == 1
    mf = retrieval_steps[0].prompt.get("metadata_filter")
    assert mf == {
        "$and": [
            {"property_id": {"$eq": "p-123"}},
            {"region": {"$eq": "san francisco"}},
        ]
    }


def test_graph_fallback_scraper_disabled_produces_skip_steps() -> None:
    """When fallback triggers but scraper is disabled, skip steps are produced."""
    matches = [
        RetrievedReview(vector_id="v1", score=0.20, metadata={
            "review_text": "Irrelevant.", "property_id": "p1", "region": "la",
        }),
    ]
    graph = _build(matches=matches)
    result = graph.invoke({"prompt": "wifi?", "context": {}, "steps": []})
    web_scrape_steps = [s for s in result["steps"] if s.module == "reviews_agent.web_scrape"]
    assert len(web_scrape_steps) == 1
    assert web_scrape_steps[0].response["status"] == "skipped"
