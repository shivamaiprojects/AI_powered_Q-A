"""Tests for the FAISS vector store."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from faq_rag.vectorstore.store import VectorStore


@pytest.fixture
def toy_store() -> VectorStore:
    vectors = np.eye(4, dtype=np.float32)
    metadata = pd.DataFrame(
        {
            "chunk_id": [f"c{i}" for i in range(4)],
            "chunk_text": [f"text {i}" for i in range(4)],
        }
    )
    return VectorStore.build(vectors, metadata)


def test_size_and_dimension(toy_store: VectorStore) -> None:
    assert toy_store.size == 4
    assert toy_store.dimension == 4


def test_search_returns_self_first(toy_store: VectorStore) -> None:
    query = np.array([1, 0, 0, 0], dtype=np.float32)
    hits = toy_store.search(query, k=2)[0]
    assert hits[0]["chunk_id"] == "c0"
    assert hits[0]["score"] == pytest.approx(1.0)


def test_length_mismatch_raises() -> None:
    vectors = np.eye(3, dtype=np.float32)
    metadata = pd.DataFrame({"chunk_id": ["a", "b"]})
    with pytest.raises(ValueError, match="mismatch"):
        VectorStore.build(vectors, metadata)


def test_save_and_load_roundtrip(toy_store: VectorStore, tmp_path) -> None:
    toy_store.save(tmp_path)
    loaded = VectorStore.load(tmp_path)
    assert loaded.size == toy_store.size
    query = np.array([0, 1, 0, 0], dtype=np.float32)
    assert loaded.search(query, k=1)[0][0]["chunk_id"] == "c1"