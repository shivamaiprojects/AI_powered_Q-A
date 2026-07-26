"""High-level retrieval over the FAISS vector store."""

from __future__ import annotations

from dataclasses import dataclass, field

from config.settings import settings
from faq_rag.embeddings.embedder import Embedder
from faq_rag.logger import get_logger
from faq_rag.vectorstore.store import VectorStore

logger = get_logger(__name__)


@dataclass
class RetrievedDocument:
    """A parent document assembled from one or more retrieved chunks."""

    doc_id: str
    question: str
    source_site: str
    score: float
    text: str
    chunk_ids: list[str] = field(default_factory=list)


class Retriever:
    """Embeds a query, searches FAISS, and returns grouped documents."""

    def __init__(
        self,
        store: VectorStore | None = None,
        embedder: Embedder | None = None,
        score_threshold: float | None = None,
        reranker=None,
    ) -> None:
        self.store = store or VectorStore.load()
        self.embedder = embedder or Embedder()
        self.score_threshold = (
            settings.retrieval_score_threshold
            if score_threshold is None
            else score_threshold
        )
        self.reranker = reranker

    def retrieve(
        self,
        query: str,
        k: int | None = None,
        site: str | None = None,
        apply_threshold: bool = True,
    ) -> list[RetrievedDocument]:
        """Return up to k parent documents most relevant to the query."""
        k = k or settings.retrieval_top_k

        fetch_k = settings.rerank_fetch_k if self.reranker is not None else k * 4
        query_vector = self.embedder.encode_query(query)
        raw_hits = self.store.search(query_vector, k=fetch_k)[0]

        if site is not None:
            raw_hits = [h for h in raw_hits if h["source_site"] == site]

        if apply_threshold:
            raw_hits = [h for h in raw_hits if h["score"] >= self.score_threshold]

        if self.reranker is not None and raw_hits:
            raw_hits = self.reranker.rerank(query, raw_hits, top_k=len(raw_hits))

        documents = self._group_by_document(raw_hits)
        return documents[:k]

    @staticmethod
    def _group_by_document(
        hits: list[dict[str, object]],
    ) -> list[RetrievedDocument]:
        """Collapse chunk hits into parent documents, keeping the best score."""
        grouped: dict[str, RetrievedDocument] = {}

        for hit in hits:
            doc_id = str(hit["doc_id"])
            score = float(hit["score"])

            if doc_id not in grouped:
                grouped[doc_id] = RetrievedDocument(
                    doc_id=doc_id,
                    question=str(hit["question"]),
                    source_site=str(hit["source_site"]),
                    score=score,
                    text=str(hit["chunk_text"]),
                    chunk_ids=[str(hit["chunk_id"])],
                )
            else:
                existing = grouped[doc_id]
                existing.chunk_ids.append(str(hit["chunk_id"]))
                existing.text = f"{existing.text}\n\n{hit['chunk_text']}"
                existing.score = max(existing.score, score)

        return sorted(grouped.values(), key=lambda d: d.score, reverse=True)