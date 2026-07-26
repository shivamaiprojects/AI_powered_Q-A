"""
Run the full evaluation suite and write results to reports/metrics/.

Run from the project root:
    python scripts/run_evaluation.py
    python scripts/run_evaluation.py --no-generation
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config.settings import settings  # noqa: E402
from faq_rag.evaluation.dataset import build_eval_set  # noqa: E402
from faq_rag.evaluation.generation_metrics import evaluate_generation  # noqa: E402
from faq_rag.evaluation.retrieval_metrics import evaluate_retrieval  # noqa: E402
from faq_rag.logger import get_logger  # noqa: E402

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the RAG system.")
    parser.add_argument("--no-generation", action="store_true")
    parser.add_argument("--gen-samples", type=int, default=50)
    args = parser.parse_args()

    eval_set = build_eval_set()

    logger.info("Evaluating retrieval on %d queries", len(eval_set))
    retrieval = evaluate_retrieval(eval_set)

    report = {
        "config": {
            "embedding_model": settings.embedding_model,
            "eval_queries": int(len(eval_set)),
            "threshold": settings.retrieval_score_threshold,
        },
        "retrieval": retrieval,
    }

    if not args.no_generation:
        logger.info("Evaluating generation on %d samples", args.gen_samples)
        start = time.perf_counter()
        report["generation"] = evaluate_generation(
            eval_set, n_samples=args.gen_samples
        )
        report["generation"]["wall_seconds"] = round(time.perf_counter() - start, 1)

    output = settings.reports_dir / "metrics" / "evaluation.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    logger.info("Wrote %s", output.relative_to(settings.project_root))


if __name__ == "__main__":
    main()