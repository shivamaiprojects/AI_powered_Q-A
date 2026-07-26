"""
Compare retrieval metrics with and without cross-encoder reranking.

Run from the project root:
    python scripts/evaluate_rerank.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from config.settings import settings  # noqa: E402
from faq_rag.embeddings.embedder import Embedder  # noqa: E402
from faq_rag.evaluation.dataset import build_eval_set  # noqa: E402
from faq_rag.logger import get_logger  # noqa: E402
from faq_rag.vectorstore.reranker import Reranker  # noqa: E402
from faq_rag.vectorstore.store import VectorStore  # noqa: E402

logger = get_logger(__name__)

K_VALUES = [1, 3, 5]
FETCH_K = settings.rerank_fetch_k


def _rank_of_truth(doc_ids: list[str], true_id: str) -> int | None:
    for rank, doc_id in enumerate(doc_ids):
        if doc_id == true_id:
            return rank
    return None


def _metrics_from_ranks(ranks: list[int | None], n: int) -> dict[str, float]:
    found = [r for r in ranks if r is not None]
    out = {"mrr": round(sum(1.0 / (r + 1) for r in found) / n, 4)}
    for k in K_VALUES:
        out[f"hit_rate@{k}"] = round(sum(1 for r in found if r < k) / n, 4)
    return out


def main() -> None:
    eval_set = build_eval_set()
    store = VectorStore.load()
    embedder = Embedder()
    reranker = Reranker()

    query_vectors = embedder.encode_queries(eval_set["question"].tolist())
    batch_hits = store.search(query_vectors, k=FETCH_K)

    baseline_ranks: list[int | None] = []
    rerank_ranks: list[int | None] = []

    for query, hits, true_id in zip(
        eval_set["question"], batch_hits, eval_set["doc_id"]
    ):
        true_id = str(true_id)

        baseline_ids = [str(h["doc_id"]) for h in hits]
        baseline_ranks.append(_rank_of_truth(baseline_ids, true_id))

        reranked = reranker.rerank(query, hits, top_k=len(hits))
        rerank_ids = [str(h["doc_id"]) for h in reranked]
        rerank_ranks.append(_rank_of_truth(rerank_ids, true_id))

    n = len(eval_set)
    report = {
        "fetch_k": FETCH_K,
        "queries": n,
        "baseline": _metrics_from_ranks(baseline_ranks, n),
        "reranked": _metrics_from_ranks(rerank_ranks, n),
    }

    b, r = report["baseline"], report["reranked"]
    report["delta"] = {
        key: round(r[key] - b[key], 4) for key in b
    }

    output = settings.reports_dir / "metrics" / "rerank_comparison.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()