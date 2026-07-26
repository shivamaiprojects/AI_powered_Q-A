"""The end-to-end RAG chain: retrieve, ground, generate."""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass, field

from config.settings import settings
from faq_rag.llm.client import LLMClient
from faq_rag.logger import get_logger
from faq_rag.rag.prompts import (
    NO_CONTEXT_REPLY,
    SYSTEM_PROMPT,
    build_user_prompt,
)
from faq_rag.vectorstore.retriever import RetrievedDocument, Retriever

logger = get_logger(__name__)


@dataclass
class RagResponse:
    """A complete answer with its supporting context and timing."""

    query: str
    answer: str
    sources: list[RetrievedDocument] = field(default_factory=list)
    retrieval_ms: float = 0.0
    generation_ms: float = 0.0
    answered: bool = True

    @property
    def total_ms(self) -> float:
        return self.retrieval_ms + self.generation_ms


class RagChain:
    """Orchestrates retrieval and grounded generation."""

    def __init__(
        self,
        retriever: Retriever | None = None,
        llm: LLMClient | None = None,
    ) -> None:
        self.retriever = retriever or Retriever()
        self.llm = llm or LLMClient()

    def _retrieve(
        self, query: str, k: int, site: str | None
    ) -> tuple[list[RetrievedDocument], float]:
        start = time.perf_counter()
        docs = self.retriever.retrieve(query, k=k, site=site)
        return docs, (time.perf_counter() - start) * 1000

    def answer(
        self,
        query: str,
        k: int | None = None,
        site: str | None = None,
    ) -> RagResponse:
        """Answer a query, returning the response with sources and timing."""
        k = k or settings.retrieval_top_k
        docs, retrieval_ms = self._retrieve(query, k, site)

        if not docs:
            logger.info("No documents above threshold for query: %r", query)
            return RagResponse(
                query=query,
                answer=NO_CONTEXT_REPLY,
                sources=[],
                retrieval_ms=retrieval_ms,
                answered=False,
            )

        user_prompt = build_user_prompt(query, docs)

        start = time.perf_counter()
        answer = self.llm.complete(SYSTEM_PROMPT, user_prompt)
        generation_ms = (time.perf_counter() - start) * 1000

        return RagResponse(
            query=query,
            answer=answer,
            sources=docs,
            retrieval_ms=retrieval_ms,
            generation_ms=generation_ms,
        )

    def stream(
        self,
        query: str,
        k: int | None = None,
        site: str | None = None,
    ) -> tuple[list[RetrievedDocument], Iterator[str]]:
        """Return sources immediately and a token stream for the answer."""
        k = k or settings.retrieval_top_k
        docs, _ = self._retrieve(query, k, site)

        if not docs:
            return [], iter([NO_CONTEXT_REPLY])

        user_prompt = build_user_prompt(query, docs)
        token_stream = self.llm.stream(SYSTEM_PROMPT, user_prompt)
        return docs, token_stream