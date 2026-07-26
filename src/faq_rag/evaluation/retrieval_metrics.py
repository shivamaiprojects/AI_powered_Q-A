"""Retrieval metric computation over a ground-truth eval set."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config.settings import settings
from faq_rag.embeddings.embedder import Embedder
from faq_rag.logger import get_logger
from faq_rag.vectorstore.store import VectorStore

logger = get_logger(__name__)


def _first_relevant_rank(retrieved_ids: list[str], true_id: str) -> int | None:
    """Return the 0-based rank of the first correct doc, or None if absent."""
    for rank, doc_id in enumerate(retrieved_ids):
        if doc_id == true_id:
            return rank
    return None


def _dcg_at_k(ranks: list[int], k: int) -> float:
    return sum(1.0 / np.log2(r + 2) for r in ranks if r < k)


def evaluate_retrieval(
    eval_set: pd.DataFrame,
    store: VectorStore | None = None,
    embedder: Embedder | None = None,
    k_values: list[int] | None = None,
    max_k: int = 10,
) -> dict[str, float]:
    """Compute Hit Rate, Recall, MRR, and NDCG at several k."""
    store = store or VectorStore.load()
    embedder = embedder or Embedder()
    k_values = k_values or settings.eval_k_values

    query_vectors = embedder.encode_queries(eval_set["question"].tolist())
    fetch_k = max(max(k_values), max_k)
    batch_hits = store.search(query_vectors, k=fetch_k)

    ranks: list[int | None] = []
    for hits, true_id in zip(batch_hits, eval_set["doc_id"]):
        retrieved_ids = [str(h["doc_id"]) for h in hits]
        ranks.append(_first_relevant_rank(retrieved_ids, str(true_id)))

    n = len(ranks)
    found = [r for r in ranks if r is not None]

    metrics: dict[str, float] = {}
    metrics["mrr"] = round(sum(1.0 / (r + 1) for r in found) / n, 4)

    for k in k_values:
        hit_rate = sum(1 for r in found if r < k) / n
        metrics[f"hit_rate@{k}"] = round(hit_rate, 4)

        ideal = n
        ndcg = _dcg_at_k(found, k) / ideal
        metrics[f"ndcg@{k}"] = round(ndcg, 4)

    metrics["queries"] = n
    metrics["not_found"] = n - len(found)
    return metrics