"""Quarantine ingestion service for scraped web reviews."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import re
from typing import Any

from pinecone import Pinecone

from app.services.embeddings import EmbeddingService
from app.services.web_review_scraper import ScrapedReview


@dataclass
class WebIngestResult:
    """Summarizes quarantine upsert outcomes for trace logging."""

    attempted: int
    upserted: int
    vector_ids: list[str]
    namespace: str


class WebReviewIngestService:
    """Embeds and upserts scraped reviews into a quarantine Pinecone namespace."""

    def __init__(
        self,
        *,
        enabled: bool,
        pinecone_api_key: str | None,
        index_name: str,
        namespace: str,
        embedding_service: EmbeddingService,
    ) -> None:
        self.enabled = enabled
        self.namespace = namespace
        self.embedding_service = embedding_service
        self._index = None
        if enabled and pinecone_api_key:
            client = Pinecone(api_key=pinecone_api_key)
            self._index = client.Index(index_name)

    @property
    def is_available(self) -> bool:
        """Return True when quarantine upsert can run safely."""

        return self.enabled and self._index is not None and self.embedding_service.is_available

    def upsert_scraped_reviews(
        self,
        *,
        reviews: list[ScrapedReview],
        context: dict[str, object],
    ) -> WebIngestResult:
        """Upsert scraped reviews to quarantine namespace with deterministic IDs."""

        if not self.is_available:
            raise RuntimeError("WebReviewIngestService is not available")
        if not reviews:
            return WebIngestResult(attempted=0, upserted=0, vector_ids=[], namespace=self.namespace)

        texts = [r.review_text for r in reviews]
        embeddings = self.embedding_service.embed_documents(texts)
        vectors: list[dict[str, Any]] = []
        vector_ids: list[str] = []

        property_id = self._context_str(context, "property_id") or "unknown"
        property_name = self._context_str(context, "property_name") or "unknown"
        city = self._context_str(context, "city") or "unknown"
        region = (self._context_str(context, "region") or "unknown").lower()
        scraped_at_utc = datetime.now(timezone.utc).isoformat()

        for review, emb in zip(reviews, embeddings):
            vector_id = self._build_vector_id(
                source=review.source,
                property_id=property_id,
                review_text=review.review_text,
                review_date=review.review_date,
                reviewer_name=review.reviewer_name,
            )
            vector_ids.append(vector_id)
            vectors.append(
                {
                    "id": vector_id,
                    "values": emb,
                    "metadata": self._clean_metadata(
                        {
                            "source_type": "web_scrape",
                            "source": review.source,
                            "source_url": review.source_url,
                            "property_id": property_id,
                            "property_name": property_name,
                            "city": city,
                            "region": region,
                            "review_date": review.review_date or "unknown",
                            "reviewer_name": review.reviewer_name or "unknown",
                            "rating": review.rating,
                            "review_text": review.review_text,
                            "scraped_at_utc": scraped_at_utc,
                        }
                    ),
                }
            )

        self._index.upsert(vectors=vectors, namespace=self.namespace)
        return WebIngestResult(
            attempted=len(reviews),
            upserted=len(vectors),
            vector_ids=vector_ids,
            namespace=self.namespace,
        )

    def _build_vector_id(
        self,
        *,
        source: str,
        property_id: str,
        review_text: str,
        review_date: str | None,
        reviewer_name: str | None,
    ) -> str:
        """Create deterministic IDs to reduce duplicate scraped inserts."""

        normalized = self._normalize(review_text)
        key = "|".join(
            [
                source.strip().lower(),
                property_id.strip().lower(),
                normalized,
                (review_date or "").strip().lower(),
                (reviewer_name or "").strip().lower(),
            ]
        )
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
        return f"web:{source}:{property_id}:{digest}"

    def _normalize(self, text: str) -> str:
        """Normalize text before hashing for stable dedupe keys."""

        return re.sub(r"\s+", " ", text.strip().lower())

    def _clean_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """Remove null metadata values to satisfy Pinecone metadata constraints."""

        cleaned: dict[str, Any] = {}
        for key, value in metadata.items():
            if value is None:
                continue
            cleaned[key] = value
        return cleaned

    def _context_str(self, context: dict[str, object], key: str) -> str | None:
        """Safely read optional string values from request context."""

        value = context.get(key)
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        return None
