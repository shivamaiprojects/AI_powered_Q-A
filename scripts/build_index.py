"""
Build and persist the FAISS index from precomputed embeddings.

Run from the project root:
    python scripts/build_index.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from config.settings import settings  # noqa: E402
from faq_rag.logger import get_logger  # noqa: E402
from faq_rag.vectorstore.store import VectorStore  # noqa: E402

logger = get_logger(__name__)


def main() -> None:
    vectors = np.load(settings.processed_data_dir / "embeddings.npy")
    chunks = pd.read_parquet(settings.processed_data_dir / "chunks.parquet")

    if len(vectors) != len(chunks):
        raise ValueError(
            f"embeddings ({len(vectors)}) and chunks ({len(chunks)}) differ; "
            "re-run build_embeddings.py after any change to chunks."
        )

    metadata = chunks[
        [
            "chunk_id",
            "doc_id",
            "chunk_index",
            "source_site",
            "question",
            "chunk_text",
            "chunk_clean",
        ]
    ]

    store = VectorStore.build(vectors, metadata)
    store.save()

    logger.info("Index ready: %d vectors, dim %d", store.size, store.dimension)


if __name__ == "__main__":
    main()