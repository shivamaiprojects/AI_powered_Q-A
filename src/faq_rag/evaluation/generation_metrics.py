"""Generation-quality and latency measurement for the RAG chain."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from faq_rag.embeddings.embedder import Embedder
from faq_rag.logger import get_logger
from faq_rag.rag.chain import RagChain

logger = get_logger(__name__)


def _faithfulness(answer: str, context: str, embedder: Embedder) -> float:
    """Cosine similarity between answer and its supporting context."""
    if not answer.strip():
        return 0.0
    vectors = embedder.encode_documents([answer, context], progress=False)
    return float(vectors[0] @ vectors[1])


def evaluate_generation(
    eval_set: pd.DataFrame,
    n_samples: int = 50,
    chain: RagChain | None = None,
    embedder: Embedder | None = None,
) -> dict[str, float]:
    """Measure latency percentiles and answer faithfulness on a subsample."""
    chain = chain or RagChain()
    embedder = embedder or Embedder()

    sample = eval_set.sample(min(n_samples, len(eval_set)), random_state=42)

    latencies: list[float] = []
    retrieval_latencies: list[float] = []
    faithfulness_scores: list[float] = []
    refusals = 0

    for question in sample["question"]:
        start = time.perf_counter()
        response = chain.answer(question)
        latencies.append((time.perf_counter() - start) * 1000)
        retrieval_latencies.append(response.retrieval_ms)

        if not response.answered:
            refusals += 1
            continue

        context = "\n\n".join(doc.text for doc in response.sources)
        faithfulness_scores.append(
            _faithfulness(response.answer, context, embedder)
        )

    latencies_arr = np.array(latencies)
    return {
        "n_samples": len(sample),
        "refusals": refusals,
        "latency_p50_ms": round(float(np.percentile(latencies_arr, 50)), 1),
        "latency_p95_ms": round(float(np.percentile(latencies_arr, 95)), 1),
        "latency_p99_ms": round(float(np.percentile(latencies_arr, 99)), 1),
        "retrieval_p50_ms": round(float(np.median(retrieval_latencies)), 1),
        "faithfulness_mean": round(float(np.mean(faithfulness_scores)), 4),
        "faithfulness_min": round(float(np.min(faithfulness_scores)), 4),
    }