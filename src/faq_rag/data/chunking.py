"""Splitting long answers into retrievable chunks."""

from __future__ import annotations

import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import settings
from faq_rag.logger import get_logger

logger = get_logger(__name__)

SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def build_splitter(
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> RecursiveCharacterTextSplitter:
    """Create the text splitter used across the project."""
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or settings.chunk_size,
        chunk_overlap=chunk_overlap or settings.chunk_overlap,
        separators=SEPARATORS,
        length_function=len,
        keep_separator=False,
    )


def _merge_short_pieces(pieces: list[str], minimum: int) -> list[str]:
    """Fold undersized pieces into a neighbour so no chunk is a fragment."""
    if len(pieces) <= 1:
        return pieces

    merged: list[str] = []
    for piece in pieces:
        if merged and len(piece) < minimum:
            merged[-1] = f"{merged[-1]} {piece}"
        else:
            merged.append(piece)

    if len(merged) > 1 and len(merged[0]) < minimum:
        merged[1] = f"{merged[0]} {merged[1]}"
        merged.pop(0)

    return merged


def chunk_documents(df: pd.DataFrame) -> pd.DataFrame:
    """Expand a document frame into one row per chunk."""
    splitter = build_splitter()
    records: list[dict[str, object]] = []

    for row in df.itertuples(index=False):
        pieces = _merge_short_pieces(
            splitter.split_text(row.answer_embed), settings.min_chunk_chars
        )
        for position, piece in enumerate(pieces):
            records.append(
                {
                    "chunk_id": f"{row.doc_id}::{position}",
                    "doc_id": row.doc_id,
                    "chunk_index": position,
                    "n_chunks": len(pieces),
                    "source_site": row.source_site,
                    "question": row.question,
                    "chunk_text": piece,
                }
            )

    chunks = pd.DataFrame.from_records(records)
    logger.info(
        "Chunked %s documents into %s chunks (mean %.2f per document)",
        f"{len(df):,}",
        f"{len(chunks):,}",
        len(chunks) / max(len(df), 1),
    )
    return chunks