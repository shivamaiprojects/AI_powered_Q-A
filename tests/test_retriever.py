"""Tests for the retriever's grouping and filtering logic."""

from __future__ import annotations

from faq_rag.vectorstore.retriever import Retriever


def _hit(doc_id, chunk_id, score, site="ai", question="q", text="t"):
    return {
        "doc_id": doc_id,
        "chunk_id": chunk_id,
        "score": score,
        "source_site": site,
        "question": question,
        "chunk_text": text,
    }


def test_grouping_collapses_same_document() -> None:
    hits = [
        _hit("d1", "d1::0", 0.9, text="first"),
        _hit("d1", "d1::1", 0.7, text="second"),
        _hit("d2", "d2::0", 0.8, text="other"),
    ]
    docs = Retriever._group_by_document(hits)
    assert len(docs) == 2
    assert docs[0].doc_id == "d1"
    assert docs[0].score == 0.9
    assert "first" in docs[0].text and "second" in docs[0].text
    assert len(docs[0].chunk_ids) == 2


def test_grouping_sorts_by_score() -> None:
    hits = [
        _hit("d1", "d1::0", 0.4),
        _hit("d2", "d2::0", 0.95),
    ]
    docs = Retriever._group_by_document(hits)
    assert [d.doc_id for d in docs] == ["d2", "d1"]