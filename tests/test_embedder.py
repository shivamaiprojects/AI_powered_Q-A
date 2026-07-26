"""Tests for the embedding wrapper."""

from __future__ import annotations

import numpy as np
import pytest

from faq_rag.embeddings.embedder import Embedder, query_prefix_for


@pytest.fixture(scope="module")
def embedder() -> Embedder:
    return Embedder()


@pytest.mark.parametrize(
    ("model_name", "expected"),
    [
        ("BAAI/bge-small-en-v1.5", "Represent this sentence for searching relevant passages: "),
        ("intfloat/e5-base-v2", "query: "),
        ("sentence-transformers/all-MiniLM-L6-v2", ""),
    ],
)
def test_query_prefix_for(model_name: str, expected: str) -> None:
    assert query_prefix_for(model_name) == expected


def test_vectors_are_unit_length(embedder: Embedder) -> None:
    vectors = embedder.encode_documents(["hello world", "another passage"], progress=False)
    norms = np.linalg.norm(vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_output_is_float32(embedder: Embedder) -> None:
    assert embedder.encode_documents(["text"], progress=False).dtype == np.float32


def test_semantic_similarity_beats_unrelated(embedder: Embedder) -> None:
    query = embedder.encode_query("how do I prevent overfitting")
    passages = embedder.encode_documents(
        [
            "Regularisation and dropout reduce overfitting in neural networks.",
            "The SQL INNER JOIN returns rows matching in both tables.",
        ],
        progress=False,
    )
    assert float(query @ passages[0]) > float(query @ passages[1])