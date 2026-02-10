"""Pinecone retrieval utilities for review vectors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pinecone import Pinecone


@dataclass
class RetrievedReview:
    """One Pinecone match normalized for agent consumption."""

    vector_id: str
    score: float
    metadata: dict[str, Any]


class PineconeRetriever:
    """Query wrapper around Pinecone index with defensive response parsing."""

    def __init__(self, *, api_key: str | None, index_name: str, namespace: str) -> None:
        self._namespace = namespace
        self._index = None
        if api_key:
            client = Pinecone(api_key=api_key)
            self._index = client.Index(index_name)

    @property
    def is_available(self) -> bool:
        """Return True if Pinecone client/index is initialized."""

        return self._index is not None

    def query(
        self,
        *,
        embedding: list[float],
        top_k: int,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[RetrievedReview]:
        """Run vector search and normalize matches across SDK response shapes."""

        if self._index is None:
            raise RuntimeError("Pinecone retriever is not configured (missing PINECONE_API_KEY)")

        response = self._index.query(
            vector=embedding,
            top_k=top_k,
            include_metadata=True,
            namespace=self._namespace,
            filter=metadata_filter,
        )

        raw_matches = getattr(response, "matches", None)
        if raw_matches is None and isinstance(response, dict):
            raw_matches = response.get("matches", [])
        raw_matches = raw_matches or []

        out: list[RetrievedReview] = []
        for raw in raw_matches:
            if isinstance(raw, dict):
                vector_id = str(raw.get("id", ""))
                score = float(raw.get("score", 0.0))
                metadata = dict(raw.get("metadata", {}) or {})
            else:
                vector_id = str(getattr(raw, "id", ""))
                score = float(getattr(raw, "score", 0.0))
                metadata = dict(getattr(raw, "metadata", {}) or {})
            out.append(RetrievedReview(vector_id=vector_id, score=score, metadata=metadata))
        return out

