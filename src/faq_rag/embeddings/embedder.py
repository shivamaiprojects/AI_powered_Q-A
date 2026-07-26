"""Sentence embedding model wrapper."""

from __future__ import annotations

import time

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from config.settings import settings
from faq_rag.logger import get_logger

logger = get_logger(__name__)

QUERY_PREFIXES: dict[str, str] = {
    "bge": settings.embedding_query_prefix,
    "e5": "query: ",
    "gte": "",
}


def resolve_device(requested: str) -> str:
    """Fall back to CPU when the requested accelerator is unavailable."""
    if requested == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable; using CPU")
        return "cpu"
    return requested


def query_prefix_for(model_name: str) -> str:
    """Return the instruction prefix a model expects on queries."""
    lowered = model_name.lower()
    for family, prefix in QUERY_PREFIXES.items():
        if family in lowered:
            return prefix
    return ""


class Embedder:
    """Encodes text into normalised dense vectors."""

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        batch_size: int | None = None,
        normalize: bool | None = None,
        query_prefix: str | None = None,
    ) -> None:
        self.model_name = model_name or settings.embedding_model
        self.device = resolve_device(device or settings.embedding_device)
        self.batch_size = batch_size or settings.embedding_batch_size
        self.normalize = (
            settings.embedding_normalize if normalize is None else normalize
        )
        self.query_prefix = (
            query_prefix_for(self.model_name) if query_prefix is None else query_prefix
        )

        started = time.perf_counter()
        self.model = SentenceTransformer(self.model_name, device=self.device)
        logger.info(
            "Loaded %s on %s (dim=%d, max_seq=%d) in %.1fs",
            self.model_name,
            self.device,
            self.dimension,
            self.model.max_seq_length,
            time.perf_counter() - started,
        )

    @property
    def dimension(self) -> int:
        getter = (
            getattr(self.model, "get_embedding_dimension", None)
            or self.model.get_sentence_embedding_dimension
        )
        return getter()

    def _encode(self, texts: list[str], progress: bool) -> np.ndarray:
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
            show_progress_bar=progress,
        )
        return vectors.astype(np.float32)

    def encode_documents(self, texts: list[str], progress: bool = True) -> np.ndarray:
        """Encode passages for indexing. No instruction prefix."""
        return self._encode(texts, progress)

    def encode_queries(self, texts: list[str], progress: bool = False) -> np.ndarray:
        """Encode search queries, applying the model's instruction prefix."""
        prefixed = [f"{self.query_prefix}{text}" for text in texts]
        return self._encode(prefixed, progress)

    def encode_query(self, text: str) -> np.ndarray:
        """Encode a single query into a 1-D vector."""
        return self.encode_queries([text])[0]