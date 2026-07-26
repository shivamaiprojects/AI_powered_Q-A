"""
Compare exact (Flat) and approximate (IVF) FAISS indexes.

Run from the project root:
    python scripts/benchmark_index.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import faiss  # noqa: E402
import numpy as np  # noqa: E402

from config.settings import settings  # noqa: E402
from faq_rag.logger import get_logger  # noqa: E402

logger = get_logger(__name__)

N_QUERIES = 1000
K = 10


def measure(index: faiss.Index, queries: np.ndarray, k: int) -> float:
    """Return median per-query search time in milliseconds."""
    timings = []
    for query in queries:
        start = time.perf_counter()
        index.search(query[np.newaxis, :], k)
        timings.append((time.perf_counter() - start) * 1000)
    return float(np.median(timings))


def recall_against_truth(
    approx: np.ndarray, truth: np.ndarray
) -> float:
    """Fraction of exact top-k neighbours also found by the approximate index."""
    overlap = [len(set(a) & set(t)) / len(t) for a, t in zip(approx, truth)]
    return float(np.mean(overlap))


def main() -> None:
    vectors = np.load(settings.processed_data_dir / "embeddings.npy").astype(np.float32)
    dim = vectors.shape[1]

    rng = np.random.default_rng(settings.random_seed)
    query_idx = rng.choice(len(vectors), size=N_QUERIES, replace=False)
    queries = vectors[query_idx]

    flat = faiss.IndexFlatIP(dim)
    flat.add(vectors)
    _, truth = flat.search(queries, K)

    n_cells = int(4 * np.sqrt(len(vectors)))
    quantizer = faiss.IndexFlatIP(dim)
    ivf = faiss.IndexIVFFlat(quantizer, dim, n_cells, faiss.METRIC_INNER_PRODUCT)
    ivf.train(vectors)
    ivf.add(vectors)

    results = {}
    results["flat"] = {
        "median_ms": round(measure(flat, queries, K), 4),
        "recall_at_10": 1.0,
    }

    for nprobe in (1, 8, 16, 32):
        ivf.nprobe = nprobe
        _, approx = ivf.search(queries, K)
        results[f"ivf_nprobe_{nprobe}"] = {
            "median_ms": round(measure(ivf, queries, K), 4),
            "recall_at_10": round(recall_against_truth(approx, truth), 4),
            "n_cells": n_cells,
        }

    output_dir = settings.reports_dir / "metrics"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "index_benchmark.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()