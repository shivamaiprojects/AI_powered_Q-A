"""Loading and basic measurement of the raw Stack Exchange corpus."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from config.settings import settings
from faq_rag.logger import get_logger

logger = get_logger(__name__)

QUESTION_COL = "title_body"
ANSWER_COL = "upvoted_answer"
SITE_COL = "source_site"

CHARS_PER_TOKEN = 4

PATTERNS: dict[str, re.Pattern[str]] = {
    "fenced_code": re.compile(r"```"),
    "inline_code": re.compile(r"`[^`\n]+`"),
    "html_entity": re.compile(r"&(?:[a-zA-Z]+|#\d+);"),
    "html_tag": re.compile(r"</?[a-zA-Z][^>]*>"),
    "url": re.compile(r"https?://"),
    "latex": re.compile(r"\$[^$\n]+\$"),
    "markdown_list": re.compile(r"(?m)^\s*[-*+]\s+"),
}


def load_raw_corpus(path: Path | None = None) -> pd.DataFrame:
    """Load the combined raw corpus from Parquet."""
    path = path or settings.raw_data_dir / "stackexchange_raw.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python scripts/download_data.py"
        )

    df = pd.read_parquet(path)
    logger.info("Loaded %s rows from %s", f"{len(df):,}", path.name)
    return df


def load_site(site: str) -> pd.DataFrame:
    """Load a single Stack Exchange site."""
    return load_raw_corpus(settings.raw_data_dir / f"{site}.parquet")


def add_length_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Append character, word, and estimated token counts."""
    df = df.copy()
    for label, column in (("question", QUESTION_COL), ("answer", ANSWER_COL)):
        text = df[column].astype(str)
        df[f"{label}_chars"] = text.str.len()
        df[f"{label}_words"] = text.str.split().str.len()
        df[f"{label}_tokens"] = (df[f"{label}_chars"] / CHARS_PER_TOKEN).round()
    return df


def add_pattern_flags(df: pd.DataFrame, column: str = ANSWER_COL) -> pd.DataFrame:
    """Append a boolean column per formatting pattern found in `column`."""
    df = df.copy()
    text = df[column].astype(str)
    for name, pattern in PATTERNS.items():
        df[f"has_{name}"] = text.str.contains(pattern, regex=True)
    return df