"""Tests for text cleaning functions."""

from __future__ import annotations

import pytest

from faq_rag.data.cleaning import (
    clean_for_embedding,
    collapse_whitespace,
    count_code_blocks,
    normalise_latex,
    strip_code,
    unescape_html,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("x &lt;- 5", "x <- 5"),
        ("a &amp;&amp; b", "a && b"),
        ("&quot;quoted&quot;", '"quoted"'),
        ("no entities", "no entities"),
    ],
)
def test_unescape_html(raw: str, expected: str) -> None:
    assert unescape_html(raw) == expected


def test_strip_indented_code() -> None:
    text = "Try this:\n\n    import numpy\n    x = 1\n\nIt works."
    assert "import numpy" not in strip_code(text)
    assert "It works." in strip_code(text)


def test_count_code_blocks_treats_run_as_one() -> None:
    text = "See:\n\n    line one\n    line two\n\nDone."
    assert count_code_blocks(text) == 1


def test_normalise_latex_keeps_greek_words() -> None:
    result = normalise_latex(r"the value $\alpha$ controls $\beta$")
    assert "alpha" in result
    assert "beta" in result
    assert "$" not in result
    assert "\\" not in result


def test_normalise_latex_strips_declarations() -> None:
    assert "DeclareMathOperator" not in normalise_latex(
        r"\DeclareMathOperator{\E}{E} then $\E[X]$"
    )


def test_collapse_whitespace() -> None:
    assert collapse_whitespace("a  \t b\n\n\n\nc") == "a b\n\nc"


def test_clean_for_embedding_is_idempotent() -> None:
    text = "Use &lt;- for assignment.\n\n    x <- 1\n\nSee $\\alpha$."
    once = clean_for_embedding(text)
    assert clean_for_embedding(once) == once