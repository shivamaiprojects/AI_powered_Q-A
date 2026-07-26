"""Prompt templates for grounded question answering."""

from __future__ import annotations

from faq_rag.vectorstore.retriever import RetrievedDocument

SYSTEM_PROMPT = (
    "You are a helpful FAQ assistant for data science, statistics, and "
    "machine learning questions. Answer the user's question using ONLY the "
    "provided context passages. Follow these rules strictly:\n"
    "- If the context does not contain enough information to answer, say so "
    "plainly. Do not use outside knowledge.\n"
    "- Be concise and direct. Do not repeat the question.\n"
    "- When helpful, refer to which context passage supports your answer.\n"
    "- Do not invent sources, citations, or facts not present in the context."
)

NO_CONTEXT_REPLY = (
    "I couldn't find anything relevant to that in the knowledge base, so I "
    "can't answer it reliably. Try rephrasing, or ask about a data science, "
    "statistics, or machine learning topic."
)


def format_context(documents: list[RetrievedDocument]) -> str:
    """Render retrieved documents into a numbered context block."""
    blocks = []
    for i, doc in enumerate(documents, start=1):
        blocks.append(
            f"[Passage {i}] (source: {doc.source_site}, "
            f"relevance: {doc.score:.2f})\n"
            f"Question: {doc.question}\n"
            f"Answer: {doc.text}"
        )
    return "\n\n".join(blocks)


def build_user_prompt(query: str, documents: list[RetrievedDocument]) -> str:
    """Assemble the user-turn prompt from a query and its context."""
    context = format_context(documents)
    return (
        f"Context passages:\n\n{context}\n\n"
        f"---\n\n"
        f"Question: {query}\n\n"
        f"Answer using only the context above."
    )