"""LLM chat service wrapper using LangChain ChatOpenAI for answer synthesis."""

from __future__ import annotations

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
        max_output_tokens: int = 180,
    ) -> None:
        self._model = model
        self._max_output_tokens = max(1, int(max_output_tokens))
        self._llm: ChatOpenAI | None = None
        if api_key and base_url:
            self._llm = ChatOpenAI(
                api_key=api_key,
                base_url=base_url,
                model=model,
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

    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        """Run one chat completion and return plain text output."""

        if self._llm is None:
            raise RuntimeError("Chat service is not configured (missing LLMOD_API_KEY or BASE_URL)")

        prompt = ChatPromptTemplate.from_messages([
            ("system", "{system}"),
            ("human", "{user}"),
        ])
        chain = prompt | self._llm
        result = chain.invoke({"system": system_prompt, "user": user_prompt})
        return (result.content or "").strip()
