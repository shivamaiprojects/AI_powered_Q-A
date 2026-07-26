"""
Measure the similarity-score distribution to set a retrieval threshold.

Run from the project root:
    python scripts/calibrate_threshold.py
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
from faq_rag.logger import get_logger  # noqa: E402

logger = get_logger(__name__)

SAMPLE_SIZE = 1000


def percentiles(values: np.ndarray, points: list[int]) -> dict[str, float]:
    return {f"p{p}": round(float(np.percentile(values, p)), 4) for p in points}


def main() -> None:
    chunks = pd.read_parquet(settings.processed_data_dir / "chunks.parquet")
    first = chunks[chunks["chunk_index"] == 0].sample(
        SAMPLE_SIZE, random_state=settings.random_seed
    )

    embedder = Embedder()
    questions = embedder.encode_queries(first["question"].tolist())
    passages = embedder.encode_documents(first["chunk_text"].tolist(), progress=False)

    positive = np.sum(questions * passages, axis=1)

    rng = np.random.default_rng(settings.random_seed)
    shuffled = rng.permutation(len(passages))
    while np.any(shuffled == np.arange(len(passages))):
        shuffled = rng.permutation(len(passages))
    negative = np.sum(questions * passages[shuffled], axis=1)

    report = {
        "model": settings.embedding_model,
        "positive": percentiles(positive, [5, 25, 50, 75, 95]),
        "negative": percentiles(negative, [5, 25, 50, 75, 95]),
        "positive_mean": round(float(positive.mean()), 4),
        "negative_mean": round(float(negative.mean()), 4),
    }

    p5_pos = report["positive"]["p5"]
    p95_neg = report["negative"]["p95"]
    suggested = round((p5_pos + p95_neg) / 2, 3)
    report["suggested_threshold"] = suggested

    output = settings.reports_dir / "metrics" / "threshold_calibration.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    logger.info(
        "positive p5=%.3f  negative p95=%.3f  suggested=%.3f",
        p5_pos, p95_neg, suggested,
    )


if __name__ == "__main__":
    main()