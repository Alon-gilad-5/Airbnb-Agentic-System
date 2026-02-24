"""Compatibility wrapper for the reviews RAG pipeline.

Delegates to the LangChain-first ReviewsPipeline defined in reviews_agent
while preserving the original build/invoke interface for existing callers
and tests.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from app.agents.reviews_agent import (
    LOW_EVIDENCE_PREFIX,
    NO_EVIDENCE_RESPONSE,
    ReviewsPipeline,
)
from app.schemas import StepLog
from app.services.pinecone_retriever import RetrievedReview

__all__ = [
    "NO_EVIDENCE_RESPONSE",
    "LOW_EVIDENCE_PREFIX",
    "ReviewsState",
    "build_reviews_graph",
]


class ReviewsState(TypedDict, total=False):
    """Pipeline state schema kept for type-annotation compatibility."""

    prompt: str
    context: dict[str, Any]
    metadata_filter: dict[str, Any] | None
    matches: list[RetrievedReview]
    web_matches: list[RetrievedReview]
    fallback_triggered: bool
    relevant_matches: list[RetrievedReview]
    evidence_count: int
    should_answer: bool
    disclaimer_prefix: str | None
    llm_answer: str
    final_answer: str
    steps: Annotated[list[StepLog], operator.add]


def build_reviews_graph(
    *,
    embedding_service: Any,
    retriever: Any,
    chat_service: Any,
    web_scraper: Any,
    web_ingest_service: Any,
    config: Any,
) -> ReviewsPipeline:
    """Build the reviews pipeline with injected services.

    Returns a ReviewsPipeline whose ``.invoke()`` method accepts and returns
    the same dict shape as the original StateGraph for full backward
    compatibility.
    """

    return ReviewsPipeline(
        embedding_service=embedding_service,
        retriever=retriever,
        chat_service=chat_service,
        web_scraper=web_scraper,
        web_ingest_service=web_ingest_service,
        config=config,
    )
