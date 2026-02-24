from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.agents.reviews_agent import ReviewsAgent, ReviewsAgentConfig
from app.services.pinecone_retriever import RetrievedReview
from app.services.web_review_ingest import WebIngestResult
from app.services.web_review_scraper import ScrapedReview


@dataclass
class _DummyEmbeddingService:
    is_available: bool = True

    def embed_query(self, text: str) -> list[float]:
        return [0.1, 0.2]


@dataclass
class _DummyRetriever:
    is_available: bool = True

    def query(
        self,
        *,
        embedding: list[float],
        top_k: int,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[RetrievedReview]:
        return []


@dataclass
class _DummyChatService:
    is_available: bool = False
    model: str = "dummy-model"

    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        return ""


class _DummyWebScraper:
    def __init__(self, *, reviews: list[ScrapedReview], meta: dict[str, Any] | None = None) -> None:
        self.is_available = True
        self._reviews = reviews
        self._meta = meta or {"status": "ok", "raw_count": len(reviews), "deduped_count": len(reviews)}

    def scrape_reviews(
        self,
        *,
        prompt: str,
        property_name: str | None,
        city: str | None,
        region: str | None,
        source_urls: dict[str, str] | None,
        max_reviews: int | None,
    ) -> tuple[list[ScrapedReview], dict[str, Any]]:
        return self._reviews, self._meta


class _DummyWebIngest:
    def __init__(self, *, available: bool = True) -> None:
        self.is_available = available
        self.calls: list[list[ScrapedReview]] = []

    def upsert_scraped_reviews(
        self,
        *,
        reviews: list[ScrapedReview],
        context: dict[str, object],
    ) -> WebIngestResult:
        self.calls.append(reviews)
        return WebIngestResult(
            attempted=len(reviews),
            upserted=len(reviews),
            vector_ids=[f"id:{i}" for i, _ in enumerate(reviews, start=1)],
            namespace="airbnb-reviews-web-quarantine",
        )


def _agent(*, scraped_reviews: list[ScrapedReview]) -> tuple[ReviewsAgent, _DummyWebIngest]:
    ingest = _DummyWebIngest()
    agent = ReviewsAgent(
        embedding_service=_DummyEmbeddingService(),
        retriever=_DummyRetriever(),
        chat_service=_DummyChatService(),
        web_scraper=_DummyWebScraper(reviews=scraped_reviews),
        web_ingest_service=ingest,
        config=ReviewsAgentConfig(
            relevance_score_threshold=0.40,
            min_lexical_relevance_for_upsert=0.15,
        ),
    )
    return agent, ingest


def test_build_metadata_filter_prefers_property_and_region() -> None:
    agent, _ = _agent(scraped_reviews=[])
    filt = agent._build_metadata_filter(
        "how is wifi",
        {"property_id": "p-123", "region": "  SAN FRANCISCO "},
    )
    assert filt == {
        "$and": [
            {"property_id": {"$eq": "p-123"}},
            {"region": {"$eq": "san francisco"}},
        ]
    }


def test_build_metadata_filter_property_only() -> None:
    agent, _ = _agent(scraped_reviews=[])
    filt = agent._build_metadata_filter("question", {"property_id": "p-9"})
    assert filt == {"property_id": {"$eq": "p-9"}}


def test_scrape_fallback_relevance_gate_uses_index_pairing() -> None:
    scraped = [
        ScrapedReview(source="google_maps", source_url="u1", review_text="Review one."),
        ScrapedReview(source="google_maps", source_url="u2", review_text="Review two."),
        ScrapedReview(source="google_maps", source_url="u3", review_text="Review three."),
    ]
    agent, ingest = _agent(scraped_reviews=scraped)

    def _mock_convert(*, scraped_reviews: list[ScrapedReview], prompt: str, context: dict[str, object]) -> list[RetrievedReview]:
        return [
            RetrievedReview(vector_id="v1", score=0.0, metadata={}),
            RetrievedReview(vector_id="v2", score=0.2, metadata={}),
            RetrievedReview(vector_id="v3", score=0.3, metadata={}),
        ]

    agent._convert_scraped_to_matches = _mock_convert  # type: ignore[method-assign]
    _, scrape_step, upsert_step = agent._scrape_fallback(prompt="wifi speed", context={"region": "oakland"})

    assert scrape_step.response["upsert_candidate_count"] == 2
    assert scrape_step.response["rejected_by_relevance_gate"] == 1
    assert upsert_step.response["status"] == "ok"
    assert upsert_step.response["upserted"] == 2
    assert len(ingest.calls) == 1
    assert ingest.calls[0] == [scraped[1], scraped[2]]


def test_scrape_fallback_all_zero_relevance_skips_upsert() -> None:
    scraped = [
        ScrapedReview(source="google_maps", source_url="u1", review_text="Review one."),
        ScrapedReview(source="google_maps", source_url="u2", review_text="Review two."),
    ]
    agent, ingest = _agent(scraped_reviews=scraped)

    def _mock_convert(*, scraped_reviews: list[ScrapedReview], prompt: str, context: dict[str, object]) -> list[RetrievedReview]:
        return [
            RetrievedReview(vector_id="v1", score=0.0, metadata={}),
            RetrievedReview(vector_id="v2", score=0.0, metadata={}),
        ]

    agent._convert_scraped_to_matches = _mock_convert  # type: ignore[method-assign]
    _, scrape_step, upsert_step = agent._scrape_fallback(prompt="parking", context={})

    assert scrape_step.response["upsert_candidate_count"] == 0
    assert scrape_step.response["rejected_by_relevance_gate"] == 2
    assert upsert_step.response["status"] == "skipped"
    assert ingest.calls == []


def test_scrape_fallback_duplicate_texts_do_not_collide() -> None:
    duplicated = "Same text with enough detail and punctuation for the test."
    scraped = [
        ScrapedReview(source="tripadvisor", source_url="u1", review_text=duplicated),
        ScrapedReview(source="tripadvisor", source_url="u2", review_text=duplicated),
    ]
    agent, ingest = _agent(scraped_reviews=scraped)

    def _mock_convert(*, scraped_reviews: list[ScrapedReview], prompt: str, context: dict[str, object]) -> list[RetrievedReview]:
        return [
            RetrievedReview(vector_id="v1", score=0.0, metadata={}),
            RetrievedReview(vector_id="v2", score=0.25, metadata={}),
        ]

    agent._convert_scraped_to_matches = _mock_convert  # type: ignore[method-assign]
    agent._scrape_fallback(prompt="cleanliness", context={})

    assert len(ingest.calls) == 1
    assert len(ingest.calls[0]) == 1
    assert ingest.calls[0][0] is scraped[1]

