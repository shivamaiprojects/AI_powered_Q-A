"""
Compare embedding models on encoding speed and retrieval ranking quality.

Run from the project root:
    python scripts/benchmark_embeddings.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from config.settings import settings  # noqa: E402
from faq_rag.embeddings.embedder import Embedder  # noqa: E402
from faq_rag.logger import get_logger  # noqa: E402

logger = get_logger(__name__)

CANDIDATES: list[tuple[str, str, str | None]] = [
    ("bge-small (prefix)", "BAAI/bge-small-en-v1.5", None),
    ("bge-small (no prefix)", "BAAI/bge-small-en-v1.5", ""),
    ("bge-base (no prefix)", "BAAI/bge-base-en-v1.5", ""),
    ("MiniLM-L6", "sentence-transformers/all-MiniLM-L6-v2", None),
]

SAMPLE_SIZE = 1500


def evaluate(
    model_name: str,
    questions: list[str],
    passages: list[str],
    query_prefix: str | None = None,
) -> dict[str, float]:
    """Measure encoding speed and in-batch retrieval ranking quality."""
    embedder = Embedder(model_name=model_name, query_prefix=query_prefix)

    started = time.perf_counter()
    passage_vectors = embedder.encode_documents(passages, progress=False)
    encode_seconds = time.perf_counter() - started

    query_started = time.perf_counter()
    query_vectors = embedder.encode_queries(questions)
    query_seconds = time.perf_counter() - query_started

    similarity = query_vectors @ passage_vectors.T
    truth = np.arange(len(questions))
    correct_scores = similarity[truth, truth]

    ranks = (similarity > correct_scores[:, None]).sum(axis=1)

    return {
        "dimension": float(embedder.dimension),
        "max_seq_length": float(embedder.model.max_seq_length),
        "passages_per_second": round(len(passages) / encode_seconds, 1),
        "query_ms_each": round(query_seconds / len(questions) * 1000, 3),
        "accuracy_at_1": round(float((ranks == 0).mean()), 4),
        "accuracy_at_5": round(float((ranks < 5).mean()), 4),
        "mrr": round(float(np.mean(1.0 / (ranks + 1))), 4),
        "matched_similarity": round(float(correct_scores.mean()), 4),
    }


def main() -> None:
    chunks = pd.read_parquet(settings.processed_data_dir / "chunks.parquet")
    first_chunks = chunks[chunks["chunk_index"] == 0]
    sample = first_chunks.sample(SAMPLE_SIZE, random_state=settings.random_seed)

    questions = sample["question"].tolist()
    passages = sample["chunk_text"].tolist()

    results = {}
    for label, model_name, prefix in CANDIDATES:
        logger.info("Benchmarking %s", label)
        results[label] = evaluate(model_name, questions, passages, prefix)

    output_dir = settings.reports_dir / "metrics"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "embedding_benchmark.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )

    print(pd.DataFrame(results).T.to_string())


if __name__ == "__main__":
    main()