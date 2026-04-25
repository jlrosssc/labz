"""
Markdown renderer — turns a list of StructuredBlock objects into clean Markdown.

Rules applied:
  - Consecutive table rows are merged into a proper GFM table
  - Consecutive code lines are merged into a fenced code block
  - Ordered lists reset numbering per group
  - URL patterns in text are left-linked  [url](url)
  - Blank lines separate non-list paragraphs
"""

from __future__ import annotations

import re
from typing import List

from .structure import StructuredBlock


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def render(blocks: List[StructuredBlock], *, detect_language: bool = True) -> str:
    """Convert structured blocks to a Markdown string."""
    if not blocks:
        return ""

    parts: List[str] = []
    i = 0
    prev_type = ""

    while i < len(blocks):
        b = blocks[i]

        # --- fenced code block (merge consecutive code blocks) ---
        if b.block_type == "code_block":
            code_lines = [b.text]
            j = i + 1
            while j < len(blocks) and blocks[j].block_type == "code_block":
                code_lines.append(blocks[j].text)
                j += 1
            lang = _guess_language("\n".join(code_lines)) if detect_language else ""
            parts.append(f"```{lang}\n" + "\n".join(code_lines) + "\n```")
            prev_type = "code_block"
            i = j
            continue

        # --- table rows (merge into GFM table) ---
        if b.block_type == "table_row":
            table_rows = [b.text]
            j = i + 1
            while j < len(blocks) and blocks[j].block_type == "table_row":
                table_rows.append(blocks[j].text)
                j += 1
            parts.append(_build_table(table_rows))
            prev_type = "table_row"
            i = j
            continue

        # --- separator ---
        if b.block_type == "separator":
            parts.append("---")
            prev_type = "separator"
            i += 1
            continue

        # --- headings ---
        if b.block_type == "heading_1":
            _maybe_blank(parts, prev_type)
            parts.append(f"# {_linkify(b.text)}")
            prev_type = "heading_1"
            i += 1
            continue

        if b.block_type == "heading_2":
            _maybe_blank(parts, prev_type)
            parts.append(f"## {_linkify(b.text)}")
            prev_type = "heading_2"
            i += 1
            continue

        if b.block_type == "heading_3":
            _maybe_blank(parts, prev_type)
            parts.append(f"### {_linkify(b.text)}")
            prev_type = "heading_3"
            i += 1
            continue

        # --- bullet list items ---
        if b.block_type == "list_item_bullet":
            indent = "  " * b.list_level
            parts.append(f"{indent}- {_linkify(b.text)}")
            prev_type = "list_item_bullet"
            i += 1
            continue

        # --- numbered list items ---
        if b.block_type == "list_item_numbered":
            indent = "  " * b.list_level
            idx = b.list_index if b.list_index is not None else 1
            parts.append(f"{indent}{idx}. {_linkify(b.text)}")
            prev_type = "list_item_numbered"
            i += 1
            continue

        # --- caption ---
        if b.block_type == "caption":
            parts.append(f"*{_linkify(b.text)}*")
            prev_type = "caption"
            i += 1
            continue

        # --- regular paragraph / unknown ---
        _maybe_blank(parts, prev_type)
        parts.append(_linkify(b.text))
        prev_type = "paragraph"
        i += 1

    return "\n\n".join(parts).strip() + "\n"


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    """
    Rough token count using the GPT/Claude rule of thumb:
    ~1 token per 4 characters for English prose.
    """
    return max(1, len(text) // 4)


def image_token_cost(width: int, height: int, *, model: str = "claude") -> int:
    """
    Estimate the token cost of sending this image raw to an AI model.

    Claude (Anthropic) charges 1 token per ~750 image bytes once encoded.
    As a rough but consistent approximation we use:
        tokens ≈ (width × height) / 750
    Adjust for JPEG compression factor (~10x) → / 7500
    """
    if model.lower().startswith("claude"):
        # Anthropic: image tokens depend on image size
        # Formula: ceil(width/32) * ceil(height/32) tiles ≈ 1750 tokens minimum
        import math
        tiles_w = math.ceil(width / 32)
        tiles_h = math.ceil(height / 32)
        return max(1, tiles_w * tiles_h)  # each tile ~ 1 token in this model
    # Generic estimate
    return max(1, (width * height) // 7500)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_URL_RE = re.compile(
    r"(https?://[^\s\)\]\,\"\']+)",
    re.IGNORECASE,
)


def _linkify(text: str) -> str:
    """Turn bare URLs into [url](url) Markdown links."""
    return _URL_RE.sub(r"[\1](\1)", text)


def _maybe_blank(parts: List[str], prev_type: str) -> None:
    """Ensure a blank line before the next block if needed."""
    # list items stay tight; blanks come naturally from join("\n\n")
    pass  # join("\n\n") already inserts blank lines between parts


_LANG_PATTERNS = [
    (re.compile(r"\bdef\s+\w+\s*\(|\bimport\s+\w+|\bclass\s+\w+:"), "python"),
    (re.compile(r"\bfunction\s+\w+\s*\(|const\s+\w+\s*=|\blet\s+\w+\s*=|\bvar\s+\w+"), "javascript"),
    (re.compile(r"#include\s*<|int\s+main\s*\("), "cpp"),
    (re.compile(r"public\s+(static|class)\s+\w+|System\.out\.print"), "java"),
    (re.compile(r"^\s*<[a-zA-Z][a-zA-Z0-9]*[\s/>]"), "html"),
    (re.compile(r"SELECT\s+.+\s+FROM\s+\w+", re.IGNORECASE), "sql"),
    (re.compile(r"\$\w+\s*=|echo\s+|->|namespace\s+\w+;"), "php"),
    (re.compile(r"\bfn\s+\w+\s*\(|\blet\s+mut\b|\bimpl\s+\w+"), "rust"),
    (re.compile(r"\bfunc\s+\w+\s*\(|\bpackage\s+\w+"), "go"),
    (re.compile(r"^\s*[a-zA-Z_]\w*\s*="), "bash"),
]


def _guess_language(code: str) -> str:
    for pattern, lang in _LANG_PATTERNS:
        if pattern.search(code):
            return lang
    return ""


def _build_table(rows: List[str]) -> str:
    """Build a GFM Markdown table from a list of raw row strings."""
    # Split each row into cells by large whitespace gaps
    parsed: List[List[str]] = []
    for row in rows:
        # Split on 2+ spaces or tab
        cells = [c.strip() for c in re.split(r"\t|  +", row) if c.strip()]
        parsed.append(cells)

    if not parsed:
        return ""

    # Normalise column count
    max_cols = max(len(r) for r in parsed)
    for r in parsed:
        while len(r) < max_cols:
            r.append("")

    # Build Markdown table
    def _row(cells: List[str]) -> str:
        return "| " + " | ".join(cells) + " |"

    lines = [_row(parsed[0])]
    lines.append("|" + "|".join([" --- "] * max_cols) + "|")
    for r in parsed[1:]:
        lines.append(_row(r))

    return "\n".join(lines)
