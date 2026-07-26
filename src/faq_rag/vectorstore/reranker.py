"""Cross-encoder reranking of retrieved candidates."""

from __future__ import annotations

from sentence_transformers import CrossEncoder

from config.settings import settings
from faq_rag.embeddings.embedder import resolve_device
from faq_rag.logger import get_logger

logger = get_logger(__name__)


class Reranker:
    """Re-scores query-document pairs with a cross-encoder."""

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name or settings.reranker_model
        self.device = resolve_device(device or settings.embedding_device)
        self.model = CrossEncoder(self.model_name, device=self.device)
        logger.info("Loaded reranker %s on %s", self.model_name, self.device)

    def rerank(
        self, query: str, documents: list[dict[str, object]], top_k: int
    ) -> list[dict[str, object]]:
        """Return documents reordered by cross-encoder relevance."""
        if not documents:
            return []

        pairs = [
            [query, str(doc.get("chunk_clean") or doc["chunk_text"])]
            for doc in documents
        ]
        scores = self.model.predict(pairs)

        for doc, score in zip(documents, scores):
            doc["rerank_score"] = float(score)

        reranked = sorted(
            documents, key=lambda d: d["rerank_score"], reverse=True
        )
        return reranked[:top_k]