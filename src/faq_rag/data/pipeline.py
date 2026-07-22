"""End-to-end preparation of the Stack Exchange corpus."""

from __future__ import annotations

import hashlib

import pandas as pd

from config.settings import settings
from faq_rag.data.chunking import chunk_documents
from faq_rag.data.cleaning import (
    clean_display,
    clean_for_embedding,
    clean_question,
    count_code_blocks,
)
from faq_rag.data.loader import ANSWER_COL, QUESTION_COL, SITE_COL, load_raw_corpus
from faq_rag.logger import get_logger

logger = get_logger(__name__)


def _document_id(question: str, answer: str) -> str:
    digest = hashlib.sha1(f"{question}||{answer}".encode("utf-8"))
    return digest.hexdigest()[:16]


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows sharing both a question and an answer."""
    before = len(df)
    df = df.drop_duplicates([QUESTION_COL, ANSWER_COL], keep="first")
    logger.info("Deduplicated: removed %s rows", f"{before - len(df):,}")
    return df


def apply_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """Produce cleaned question, display answer, and embedding answer."""
    df = df.copy()
    df["question"] = df[QUESTION_COL].map(clean_question)
    df["answer_clean"] = df[ANSWER_COL].map(clean_display)
    df["answer_embed"] = df[ANSWER_COL].map(clean_for_embedding)
    df["n_code_blocks"] = df[ANSWER_COL].map(count_code_blocks)
    return df


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Remove documents outside the quality thresholds."""
    lengths = df["answer_clean"].str.len()
    embed_lengths = df["answer_embed"].str.len()

    mask = (
        lengths.between(settings.min_answer_chars, settings.max_answer_chars)
        & (embed_lengths >= 100)
        & (df["question"].str.len() >= 20)
    )

    logger.info(
        "Filtered: kept %s of %s rows (%.1f%%)",
        f"{int(mask.sum()):,}",
        f"{len(df):,}",
        mask.mean() * 100,
    )
    return df[mask]


def balance_sites(df: pd.DataFrame) -> pd.DataFrame:
    """Take an equal-sized random sample from each source site."""
    target = settings.docs_per_site
    frames = [
        group.sample(min(len(group), target), random_state=settings.random_seed)
        for _, group in df.groupby(SITE_COL, sort=False)
    ]
    sampled = pd.concat(frames, ignore_index=True)
    logger.info("Balanced sites:\n%s", sampled[SITE_COL].value_counts().to_string())
    return sampled


def build_documents(df: pd.DataFrame) -> pd.DataFrame:
    """Assemble the final document table."""
    df = df.copy()
    df["doc_id"] = [
        _document_id(q, a) for q, a in zip(df["question"], df["answer_clean"])
    ]
    columns = [
        "doc_id",
        "source_site",
        "question",
        "answer_clean",
        "answer_embed",
        "n_code_blocks",
    ]
    return df[columns].reset_index(drop=True)


def run_pipeline() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Execute every preparation step and return documents and chunks."""
    raw = load_raw_corpus()
    cleaned = apply_cleaning(deduplicate(raw))
    filtered = apply_filters(cleaned)
    balanced = balance_sites(filtered)
    documents = build_documents(balanced)
    chunks = chunk_documents(documents)
    return documents, chunks