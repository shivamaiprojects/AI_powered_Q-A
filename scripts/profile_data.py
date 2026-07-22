"""
Profile the raw corpus and write summary statistics to reports/metrics/.

Run from the project root:
    python scripts/profile_data.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from config.settings import settings  # noqa: E402
from faq_rag.data.loader import (  # noqa: E402
    ANSWER_COL,
    PATTERNS,
    QUESTION_COL,
    SITE_COL,
    add_length_columns,
    add_pattern_flags,
    load_raw_corpus,
)
from faq_rag.logger import get_logger  # noqa: E402

logger = get_logger(__name__)

PERCENTILES = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]


def length_profile(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    columns = [c for c in df.columns if c.endswith(("_chars", "_words", "_tokens"))]
    described = df[columns].describe(percentiles=PERCENTILES)
    return {col: described[col].round(2).to_dict() for col in columns}


def integrity_profile(df: pd.DataFrame) -> dict[str, int]:
    return {
        "total_rows": int(len(df)),
        "null_questions": int(df[QUESTION_COL].isna().sum()),
        "null_answers": int(df[ANSWER_COL].isna().sum()),
        "duplicate_questions": int(df[QUESTION_COL].duplicated().sum()),
        "duplicate_answers": int(df[ANSWER_COL].duplicated().sum()),
        "duplicate_pairs": int(df.duplicated([QUESTION_COL, ANSWER_COL]).sum()),
    }


def pattern_profile(df: pd.DataFrame) -> dict[str, float]:
    flags = [f"has_{name}" for name in PATTERNS]
    return {name: round(float(df[name].mean() * 100), 2) for name in flags}


def main() -> None:
    df = add_pattern_flags(add_length_columns(load_raw_corpus()))

    profile = {
        "dataset": settings.dataset_name,
        "configs": settings.dataset_configs,
        "integrity": integrity_profile(df),
        "site_counts": df[SITE_COL].value_counts().to_dict(),
        "lengths": length_profile(df),
        "pattern_pct": pattern_profile(df),
        "pattern_pct_by_site": {
            site: pattern_profile(group)
            for site, group in df.groupby(SITE_COL)
        },
    }

    output_dir = settings.reports_dir / "metrics"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "data_profile.json"
    output_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")

    logger.info("Profile written to %s", output_path.relative_to(settings.project_root))
    print(json.dumps(profile["integrity"], indent=2))
    print(json.dumps(profile["pattern_pct"], indent=2))


if __name__ == "__main__":
    main()