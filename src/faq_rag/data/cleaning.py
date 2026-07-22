"""Text cleaning and normalisation for the Stack Exchange corpus."""

from __future__ import annotations

import html
import re

CODE_PLACEHOLDER = " [code] "

FENCED_CODE = re.compile(r"```.*?```", re.DOTALL)
INDENTED_CODE = re.compile(r"(?m)^(?: {4}|\t).*(?:\n(?: {4}|\t).*)*")
INLINE_CODE = re.compile(r"`[^`\n]+`")

DISPLAY_MATH = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
INLINE_MATH = re.compile(r"\$([^$\n]+?)\$")
LATEX_DECLARATION = re.compile(r"\\(?:DeclareMathOperator|newcommand|renewcommand)\s*\{[^}]*\}\s*\{[^}]*\}")
LATEX_COMMAND = re.compile(r"\\([a-zA-Z]+)")

GREEK_LETTERS = frozenset(
    "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi "
    "pi rho sigma tau upsilon phi chi psi omega".split()
)

LATEX_TRANSLATIONS = {
    "sum": "sum",
    "prod": "product",
    "int": "integral",
    "infty": "infinity",
    "partial": "partial",
    "nabla": "gradient",
    "sqrt": "square root",
    "frac": "over",
    "cdot": "times",
    "times": "times",
    "approx": "approximately",
    "sim": "distributed as",
    "neq": "not equal",
    "leq": "at most",
    "geq": "at least",
    "forall": "for all",
    "exists": "exists",
    "hat": "estimated",
    "bar": "mean",
    "log": "log",
    "exp": "exp",
    "min": "min",
    "max": "max",
    "argmin": "argmin",
    "argmax": "argmax",
}

WHITESPACE = re.compile(r"[ \t]+")
BLANK_LINES = re.compile(r"\n{3,}")
URL = re.compile(r"https?://\S+")


def unescape_html(text: str) -> str:
    """Convert HTML entities to their literal characters, twice if needed."""
    once = html.unescape(text)
    return html.unescape(once) if "&" in once else once


def count_code_blocks(text: str) -> int:
    """Count fenced and indented code regions."""
    return len(FENCED_CODE.findall(text)) + len(INDENTED_CODE.findall(text))


def strip_code(text: str) -> str:
    """Replace code regions with a neutral placeholder."""
    text = FENCED_CODE.sub(CODE_PLACEHOLDER, text)
    text = INDENTED_CODE.sub(CODE_PLACEHOLDER, text)
    return INLINE_CODE.sub(CODE_PLACEHOLDER, text)


def _translate_commands(expression: str) -> str:
    def replace(match: re.Match[str]) -> str:
        command = match.group(1)
        if command in GREEK_LETTERS:
            return f" {command} "
        return f" {LATEX_TRANSLATIONS.get(command, '')} "

    return LATEX_COMMAND.sub(replace, expression)


def normalise_latex(text: str) -> str:
    """Rewrite LaTeX expressions as plain words the embedder can use."""
    text = LATEX_DECLARATION.sub(" ", text)
    text = DISPLAY_MATH.sub(lambda m: f" {_translate_commands(m.group(1))} ", text)
    text = INLINE_MATH.sub(lambda m: f" {_translate_commands(m.group(1))} ", text)
    text = _translate_commands(text)
    return text.replace("$", " ").replace("{", " ").replace("}", " ")


def collapse_whitespace(text: str) -> str:
    """Normalise runs of spaces and blank lines."""
    text = WHITESPACE.sub(" ", text)
    text = BLANK_LINES.sub("\n\n", text)
    return "\n".join(line.strip() for line in text.split("\n")).strip()


def clean_display(text: str) -> str:
    """Produce the answer text shown to users and passed to the LLM."""
    return collapse_whitespace(unescape_html(text))


def clean_for_embedding(text: str) -> str:
    """Produce the answer text used to build vectors."""
    text = unescape_html(text)
    text = strip_code(text)
    text = normalise_latex(text)
    text = URL.sub(" ", text)
    return collapse_whitespace(text)


def clean_question(text: str) -> str:
    """Clean a question title."""
    return collapse_whitespace(unescape_html(text))