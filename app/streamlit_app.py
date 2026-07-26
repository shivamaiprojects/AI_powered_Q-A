"""Streamlit interface for the RAG FAQ assistant."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import streamlit as st

from config.settings import settings
from faq_rag.rag.chain import RagChain

st.set_page_config(
    page_title="FAQ Assistant",
    page_icon="◆",
    layout="wide",
)

CSS = """
<style>
    .stApp { background: #0f1115; }
    h1, h2, h3, p, label, span, div { color: #e4e7eb; }
    .block-container { padding-top: 3rem; max-width: 1100px; }

    .eyebrow {
        font-family: 'SF Mono', 'Roboto Mono', monospace;
        font-size: 0.75rem;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        color: #f0a933;
        margin-bottom: 0.5rem;
    }
    .headline {
        font-size: 2rem;
        font-weight: 700;
        line-height: 1.15;
        margin-bottom: 0.25rem;
    }
    .subhead {
        color: #8b93a1;
        font-size: 0.95rem;
        margin-bottom: 2rem;
    }
    .source-card {
        background: #171a21;
        border: 1px solid #262b35;
        border-left: 3px solid #f0a933;
        border-radius: 6px;
        padding: 0.85rem 1rem;
        margin-bottom: 0.75rem;
    }
    .source-meta {
        font-family: 'SF Mono', 'Roboto Mono', monospace;
        font-size: 0.72rem;
        color: #8b93a1;
        display: flex;
        justify-content: space-between;
        margin-bottom: 0.4rem;
    }
    .source-score { color: #f0a933; }
    .source-question { font-size: 0.9rem; font-weight: 600; color: #e4e7eb; }
    .refusal {
        background: #1a1512;
        border: 1px solid #3d2f1a;
        border-radius: 6px;
        padding: 1rem 1.25rem;
        color: #d4a76a;
    }
    .timing {
        font-family: 'SF Mono', 'Roboto Mono', monospace;
        font-size: 0.72rem;
        color: #5a616e;
        margin-top: 1rem;
    }
    div[data-testid="stTextInput"] input {
        background: #171a21;
        color: #e4e7eb;
        border: 1px solid #262b35;
        font-size: 1rem;
    }
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource
def load_chain() -> RagChain:
    return RagChain()


@st.cache_data
def load_sites() -> list[str]:
    return RagChain().retriever.available_sites()


chain = load_chain()

st.markdown('<div class="eyebrow">Retrieval-Augmented Q&A</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="headline">Ask the knowledge base.</div>', unsafe_allow_html=True
)
st.markdown(
    '<div class="subhead">Grounded answers from a 5,100-document corpus of '
    "data science, statistics, and machine learning discussion. Every answer "
    "shows the passages it was built from.</div>",
    unsafe_allow_html=True,
)

col_query, col_filter = st.columns([4, 1])
with col_query:
    query = st.text_input(
        "Your question",
        placeholder="e.g. how do I stop my model overfitting?",
        label_visibility="collapsed",
    )
with col_filter:
    site = st.selectbox(
        "Source",
        options=["all", *load_sites()],
        label_visibility="collapsed",
    )

if query:
    site_filter = None if site == "all" else site
    answer_col, source_col = st.columns([3, 2], gap="large")

    with answer_col:
        st.markdown("#### Answer")
        sources, token_stream = chain.stream(query, site=site_filter)

        if not sources:
            st.markdown(
                '<div class="refusal">No passages in the knowledge base cleared '
                "the relevance threshold for this question. Try rephrasing, or "
                "ask about a data science, statistics, or ML topic.</div>",
                unsafe_allow_html=True,
            )
        else:
            st.write_stream(token_stream)

    with source_col:
        if sources:
            st.markdown("#### Sources")
            for i, doc in enumerate(sources, start=1):
                st.markdown(
                    f'<div class="source-card">'
                    f'<div class="source-meta">'
                    f"<span>[{i}] {doc.source_site}</span>"
                    f'<span class="source-score">{doc.score:.2f}</span>'
                    f"</div>"
                    f'<div class="source-question">{doc.question}</div>'
                    f"</div>",
                    unsafe_allow_html=True,
                )