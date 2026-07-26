"""Tests for cross-encoder reranking logic."""

from __future__ import annotations

from faq_rag.vectorstore.reranker import Reranker


class StubModel:
    def predict(self, pairs):
        return [len(p[1]) for p in pairs]


def test_rerank_orders_by_score(monkeypatch) -> None:
    reranker = Reranker.__new__(Reranker)
    reranker.model = StubModel()

    docs = [
        {"chunk_text": "short", "doc_id": "a"},
        {"chunk_text": "much longer text here", "doc_id": "b"},
    ]
    result = reranker.rerank("query", docs, top_k=2)
    assert result[0]["doc_id"] == "b"
    assert "rerank_score" in result[0]


def test_rerank_empty_returns_empty() -> None:
    reranker = Reranker.__new__(Reranker)
    reranker.model = None
    assert reranker.rerank("query", [], top_k=5) == []