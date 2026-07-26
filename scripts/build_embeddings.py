"""
Encode all chunks and persist the vectors.

Run from the project root:
    python scripts/build_embeddings.py
"""

from __future__ import annotations

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


def main() -> None:
    chunks = pd.read_parquet(settings.processed_data_dir / "chunks.parquet")
    texts = chunks["chunk_text"].tolist()

    embedder = Embedder()

    started = time.perf_counter()
    vectors = embedder.encode_documents(texts)
    elapsed = time.perf_counter() - started

    output = settings.processed_data_dir / "embeddings.npy"
    np.save(output, vectors)

    logger.info(
        "Encoded %s chunks in %.1fs (%.0f chunks/s) -> shape %s, %.1f MB",
        f"{len(texts):,}",
        elapsed,
        len(texts) / elapsed,
        vectors.shape,
        vectors.nbytes / 1e6,
    )

    norms = np.linalg.norm(vectors, axis=1)
    logger.info("Vector norms: min=%.4f max=%.4f", norms.min(), norms.max())


if __name__ == "__main__":
    main()