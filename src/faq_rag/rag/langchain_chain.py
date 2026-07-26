"""LangChain-native assembly of the RAG pipeline using LCEL."""

from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI

from config.settings import settings
from faq_rag.rag.prompts import SYSTEM_PROMPT, format_context
from faq_rag.vectorstore.retriever import Retriever


def build_langchain_rag():
    """Construct the RAG pipeline as an LCEL runnable."""
    retriever = Retriever()

    llm = ChatOpenAI(
        model=settings.llm_model,
        base_url=settings.llm_base_url,
        api_key=settings.hf_token,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            (
                "user",
                "Context passages:\n\n{context}\n\n---\n\n"
                "Question: {question}\n\nAnswer using only the context above.",
            ),
        ]
    )

    def retrieve_context(query: str) -> str:
        docs = retriever.retrieve(query)
        return format_context(docs)

    chain = (
        {
            "context": RunnableLambda(retrieve_context),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain