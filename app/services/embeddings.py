"""Embedding service backed by LLMOD/Azure-compatible embeddings API."""

from __future__ import annotations

from langchain_openai import AzureOpenAIEmbeddings


class EmbeddingService:
    """Small wrapper that keeps embedding initialization and error surface simple."""

    def __init__(
        self,
        *,
        api_key: str | None,
        azure_endpoint: str | None,
        model: str,
        deployment: str,
    ) -> None:
        self._client: AzureOpenAIEmbeddings | None = None
        if api_key and azure_endpoint:
            self._client = AzureOpenAIEmbeddings(
                model=model,
                azure_deployment=deployment,
                azure_endpoint=azure_endpoint,
                api_key=api_key,
                check_embedding_ctx_length=False,
            )

    @property
    def is_available(self) -> bool:
        """Return True when embedding calls can be executed."""

        return self._client is not None

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string for vector search."""

        if self._client is None:
            raise RuntimeError("Embedding service is not configured (missing API key or BASE_URL)")
        return self._client.embed_query(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple documents for bulk vector upsert workflows."""

        if self._client is None:
            raise RuntimeError("Embedding service is not configured (missing API key or BASE_URL)")
        return self._client.embed_documents(texts)
