"""LLM chat service wrapper using LangChain ChatOpenAI for answer synthesis."""

from __future__ import annotations

import httpx
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


class ChatService:
    """Encapsulates chat-completion calls and availability checks."""

    def __init__(
        self,
        *,
        api_key: str | None,
        base_url: str | None,
        model: str,
        provider_name: str = "llmod",
        default_headers: dict[str, str] | None = None,
        max_output_tokens: int = 180,
    ) -> None:
        self._provider_name = provider_name.strip().lower() or "llmod"
        self._model = model
        self._max_output_tokens = max(1, int(max_output_tokens))
        self._llm: ChatOpenAI | None = None
        self._http_client: httpx.Client | None = None
        if api_key and base_url:
            # Ignore process-wide proxy env vars so local dev shells with
            # placeholder proxy settings do not break outbound model calls.
            self._http_client = httpx.Client(trust_env=False)
            self._llm = ChatOpenAI(
                api_key=api_key,
                base_url=base_url,
                model=model,
                default_headers=default_headers,
                http_client=self._http_client,
                max_tokens=self._max_output_tokens,
                temperature=1 if "gpt-5" in model.lower() else 0.2,
                max_retries=0,
            )

    @property
    def is_available(self) -> bool:
        """Return True when chat generation can be executed."""

        return self._llm is not None

    @property
    def model(self) -> str:
        """Expose the configured model name for step logging."""

        return self._model

    @property
    def provider_name(self) -> str:
        """Expose provider identifier for runtime diagnostics."""

        return self._provider_name

    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        """Run one chat completion and return plain text output."""

        if self._llm is None:
            raise RuntimeError(
                f"Chat service '{self._provider_name}' is not configured "
                "(missing provider API key/base URL)"
            )

        prompt = ChatPromptTemplate.from_messages([
            ("system", "{system}"),
            ("human", "{user}"),
        ])
        chain = prompt | self._llm
        result = chain.invoke({"system": system_prompt, "user": user_prompt})
        return (result.content or "").strip()
