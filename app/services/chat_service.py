"""LLM chat service wrapper used by agents for answer synthesis."""

from __future__ import annotations

from openai import OpenAI


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
        self._client: OpenAI | None = None
        if api_key and base_url:
            self._client = OpenAI(api_key=api_key, base_url=base_url)

    @property
    def is_available(self) -> bool:
        """Return True when chat generation can be executed."""

        return self._client is not None

    @property
    def model(self) -> str:
        """Expose the configured model name for step logging."""

        return self._model

    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        """Run one chat completion and return plain text output."""

        if self._client is None:
            raise RuntimeError("Chat service is not configured (missing LLMOD_API_KEY or BASE_URL)")

        params = {
            "model": self._model,
            "max_tokens": self._max_output_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        # Some LLMOD gpt-5 model groups only accept temperature=1.
        if "gpt-5" in self._model.lower():
            params["temperature"] = 1
        else:
            params["temperature"] = 0.2

        response = self._client.chat.completions.create(**params)
        return (response.choices[0].message.content or "").strip()
