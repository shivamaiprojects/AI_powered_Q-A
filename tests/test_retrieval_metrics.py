"""Tests for retrieval metric math on hand-constructed rankings."""

from __future__ import annotations

import pandas as pd

from faq_rag.evaluation.retrieval_metrics import (
    _dcg_at_k,
    _first_relevant_rank,
)


def test_first_relevant_rank_found() -> None:
    assert _first_relevant_rank(["a", "b", "c"], "b") == 1


def test_first_relevant_rank_absent() -> None:
    assert _first_relevant_rank(["a", "b", "c"], "z") is None


def test_first_relevant_rank_first_position() -> None:
    assert _first_relevant_rank(["x", "y"], "x") == 0


def test_dcg_rewards_higher_ranks() -> None:
    top = _dcg_at_k([0], 5)
    low = _dcg_at_k([4], 5)
    assert top > low
    assert top == 1.0


def test_dcg_respects_cutoff() -> None:
    assert _dcg_at_k([7], 5) == 0.0