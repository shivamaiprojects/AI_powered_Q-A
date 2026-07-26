"""LLM client over the OpenAI-compatible HF router."""

from __future__ import annotations

from collections.abc import Iterator

from openai import OpenAI

from config.settings import settings
from faq_rag.logger import get_logger

logger = get_logger(__name__)


class LLMClient:
    """Thin wrapper over an OpenAI-compatible chat completion endpoint."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        self.model = model or settings.llm_model
        self.temperature = (
            settings.llm_temperature if temperature is None else temperature
        )
        self.max_tokens = max_tokens or settings.llm_max_tokens
        self._client = OpenAI(
            base_url=base_url or settings.llm_base_url,
            api_key=api_key or settings.llm_api_key,
            timeout=settings.llm_timeout,
        )

    def _messages(self, system_prompt: str, user_prompt: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Return a single completion for a system + user message pair."""
        response = self._client.chat.completions.create(
            model=self.model,
            messages=self._messages(system_prompt, user_prompt),
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content or ""

    def stream(self, system_prompt: str, user_prompt: str) -> Iterator[str]:
        """Yield completion text incrementally as it is generated."""
        stream = self._client.chat.completions.create(
            model=self.model,
            messages=self._messages(system_prompt, user_prompt),
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta