"""Build a ground-truth evaluation set from the chunk corpus."""

from __future__ import annotations

import pandas as pd

from config.settings import settings
from faq_rag.logger import get_logger

logger = get_logger(__name__)


def build_eval_set(sample_size: int | None = None) -> pd.DataFrame:
    """Sample question/doc_id pairs to use as retrieval ground truth."""
    sample_size = sample_size or settings.eval_sample_size
    chunks = pd.read_parquet(settings.processed_data_dir / "chunks.parquet")

    first_chunks = chunks[chunks["chunk_index"] == 0].copy()

    per_site = max(sample_size // first_chunks["source_site"].nunique(), 1)
    frames = [
        group.sample(min(len(group), per_site), random_state=settings.random_seed)
        for _, group in first_chunks.groupby("source_site", sort=False)
    ]
    eval_set = pd.concat(frames, ignore_index=True)

    eval_set = eval_set[["question", "doc_id", "source_site"]]
    logger.info(
        "Built eval set: %d queries across %d sites",
        len(eval_set),
        eval_set["source_site"].nunique(),
    )
    return eval_set