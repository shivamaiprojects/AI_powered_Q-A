"""Tests for the RAG chain orchestration."""

from __future__ import annotations

from faq_rag.rag.chain import RagChain, RagResponse
from faq_rag.rag.prompts import NO_CONTEXT_REPLY
from faq_rag.vectorstore.retriever import RetrievedDocument


class StubRetriever:
    def __init__(self, docs):
        self._docs = docs

    def retrieve(self, query, k=None, site=None):
        return self._docs


class StubLLM:
    def __init__(self):
        self.called = False

    def complete(self, system_prompt, user_prompt):
        self.called = True
        return "stub answer"


def _doc():
    return RetrievedDocument(
        doc_id="d1", question="q", source_site="ai",
        score=0.8, text="context text",
    )


def test_answer_uses_llm_when_docs_present() -> None:
    llm = StubLLM()
    chain = RagChain(retriever=StubRetriever([_doc()]), llm=llm)
    response = chain.answer("real question")
    assert response.answered is True
    assert response.answer == "stub answer"
    assert llm.called is True
    assert len(response.sources) == 1


def test_answer_short_circuits_when_no_docs() -> None:
    llm = StubLLM()
    chain = RagChain(retriever=StubRetriever([]), llm=llm)
    response = chain.answer("gibberish")
    assert response.answered is False
    assert response.answer == NO_CONTEXT_REPLY
    assert llm.called is False


def test_total_ms_sums_components() -> None:
    r = RagResponse(query="q", answer="a", retrieval_ms=5.0, generation_ms=45.0)
    assert r.total_ms == 50.0