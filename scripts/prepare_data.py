"""
Clean, filter, balance, and chunk the raw corpus.

Run from the project root:
    python scripts/prepare_data.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from config.settings import settings  # noqa: E402
from faq_rag.data.pipeline import run_pipeline  # noqa: E402
from faq_rag.logger import get_logger  # noqa: E402

logger = get_logger(__name__)


def summarise(documents: pd.DataFrame, chunks: pd.DataFrame) -> dict[str, object]:
    return {
        "n_documents": int(len(documents)),
        "n_chunks": int(len(chunks)),
        "chunks_per_document": round(len(chunks) / len(documents), 2),
        "documents_per_site": documents["source_site"].value_counts().to_dict(),
        "answer_clean_chars": documents["answer_clean"].str.len().describe().round(1).to_dict(),
        "chunk_chars": chunks["chunk_text"].str.len().describe().round(1).to_dict(),
        "documents_with_code": int((documents["n_code_blocks"] > 0).sum()),
        "config": {
            "min_answer_chars": settings.min_answer_chars,
            "max_answer_chars": settings.max_answer_chars,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "docs_per_site": settings.docs_per_site,
        },
    }


def main() -> None:
    documents, chunks = run_pipeline()

    settings.processed_data_dir.mkdir(parents=True, exist_ok=True)
    documents.to_parquet(settings.processed_data_dir / "documents.parquet", index=False)
    chunks.to_parquet(settings.processed_data_dir / "chunks.parquet", index=False)

    profile = summarise(documents, chunks)
    metrics_dir = settings.reports_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "processed_profile.json").write_text(
        json.dumps(profile, indent=2, default=float), encoding="utf-8"
    )

    logger.info("Wrote documents and chunks to %s", settings.processed_data_dir.name)
    print(json.dumps(profile, indent=2, default=float))


if __name__ == "__main__":
    main()