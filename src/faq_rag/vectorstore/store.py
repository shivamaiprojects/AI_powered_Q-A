"""FAISS vector store: build, search, persist, load."""

from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np
import pandas as pd

from config.settings import settings
from faq_rag.logger import get_logger

logger = get_logger(__name__)

INDEX_FILENAME = "index.faiss"
METADATA_FILENAME = "metadata.parquet"


class VectorStore:
    """A FAISS index paired with per-vector chunk metadata."""

    def __init__(self, index: faiss.Index, metadata: pd.DataFrame) -> None:
        self.index = index
        self.metadata = metadata.reset_index(drop=True)

    @property
    def size(self) -> int:
        return self.index.ntotal

    @property
    def dimension(self) -> int:
        return self.index.d

    @classmethod
    def build(cls, vectors: np.ndarray, metadata: pd.DataFrame) -> "VectorStore":
        """Construct an exact inner-product index from normalised vectors."""
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)
        if len(vectors) != len(metadata):
            raise ValueError(
                f"vector/metadata length mismatch: {len(vectors)} vs {len(metadata)}"
            )

        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        logger.info(
            "Built IndexFlatIP: %d vectors, dim %d", index.ntotal, index.d
        )
        return cls(index, metadata)

    def search(
        self, query_vectors: np.ndarray, k: int | None = None
    ) -> list[list[dict[str, object]]]:
        """Return the top-k metadata records for each query vector."""
        k = k or settings.retrieval_top_k
        if query_vectors.ndim == 1:
            query_vectors = query_vectors[np.newaxis, :]
        if query_vectors.dtype != np.float32:
            query_vectors = query_vectors.astype(np.float32)

        scores, indices = self.index.search(query_vectors, k)

        results: list[list[dict[str, object]]] = []
        for row_scores, row_indices in zip(scores, indices):
            hits = []
            for score, idx in zip(row_scores, row_indices):
                if idx == -1:
                    continue
                record = self.metadata.iloc[int(idx)].to_dict()
                record["score"] = float(score)
                hits.append(record)
            results.append(hits)
        return results

    def save(self, directory: Path | None = None) -> Path:
        """Persist the index and metadata to disk."""
        directory = directory or settings.vectorstore_dir
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / INDEX_FILENAME))
        self.metadata.to_parquet(directory / METADATA_FILENAME, index=False)
        logger.info("Saved vector store to %s", directory)
        return directory

    @classmethod
    def load(cls, directory: Path | None = None) -> "VectorStore":
        """Load a persisted vector store from disk."""
        directory = directory or settings.vectorstore_dir
        index_path = directory / INDEX_FILENAME
        metadata_path = directory / METADATA_FILENAME
        if not index_path.exists():
            raise FileNotFoundError(
                f"{index_path} not found. Run: python scripts/build_index.py"
            )
        index = faiss.read_index(str(index_path))
        metadata = pd.read_parquet(metadata_path)
        logger.info("Loaded vector store: %d vectors", index.ntotal)
        return cls(index, metadata)