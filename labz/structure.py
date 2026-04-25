"""
Structure detection — classifies OCR paragraphs into semantic block types:

  heading_1 / heading_2 / heading_3
  paragraph
  list_item (bullet or numbered)
  code_block
  table_row
  caption
  separator
  unknown

The algorithm is purely heuristic (no ML) so it works 100% offline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

from .ocr import OcrResult, Paragraph


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

BLOCK_TYPES = (
    "heading_1",
    "heading_2",
    "heading_3",
    "paragraph",
    "list_item_bullet",
    "list_item_numbered",
    "code_block",
    "table_row",
    "caption",
    "separator",
    "unknown",
)


@dataclass
class StructuredBlock:
    text: str
    block_type: str
    list_level: int = 0          # indentation depth for list items
    list_index: Optional[int] = None  # numeric index for ordered lists
    left: int = 0
    top: int = 0
    avg_char_height: float = 12.0
    source_paragraph: Optional[Paragraph] = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify(ocr: OcrResult) -> List[StructuredBlock]:
    """
    Take an :class:`OcrResult` and return a list of :class:`StructuredBlock`
    objects sorted top-to-bottom.
    """
    if not ocr.paragraphs:
        return []

    base_height = ocr.global_avg_word_height or 12.0
    blocks: List[StructuredBlock] = []

    for para in ocr.paragraphs:
        text = _clean_text(para.text)
        if not text:
            continue

        avg_h = para.avg_line_height or base_height
        ratio = avg_h / base_height if base_height else 1.0

        sb = StructuredBlock(
            text=text,
            block_type="unknown",
            left=para.left,
            top=para.top,
            avg_char_height=avg_h,
            source_paragraph=para,
        )

        # ---- heading detection ----
        if _is_heading(text, ratio, para):
            if ratio >= 1.8:
                sb.block_type = "heading_1"
            elif ratio >= 1.4:
                sb.block_type = "heading_2"
            else:
                sb.block_type = "heading_3"

        # ---- code block ----
        elif _is_code(text, para):
            sb.block_type = "code_block"

        # ---- list: bullet ----
        elif _is_bullet_list(text):
            sb.block_type = "list_item_bullet"
            sb.list_level = _indent_level(para, ocr)
            sb.text = _strip_bullet(text)

        # ---- list: numbered ----
        elif _is_numbered_list(text):
            sb.block_type = "list_item_numbered"
            sb.list_level = _indent_level(para, ocr)
            sb.list_index, sb.text = _parse_numbered(text)

        # ---- separator / horizontal rule ----
        elif _is_separator(text):
            sb.block_type = "separator"

        # ---- table row ----
        elif _is_table_row(text, para):
            sb.block_type = "table_row"

        # ---- caption (short, italic-ish) ----
        elif _is_caption(text, para):
            sb.block_type = "caption"

        # ---- regular paragraph ----
        else:
            sb.block_type = "paragraph"

        blocks.append(sb)

    return blocks


# ---------------------------------------------------------------------------
# Heuristics
# ---------------------------------------------------------------------------

_BULLET_RE = re.compile(
    r"^[\u2022\u2023\u25E6\u2043\u2219\-\*\+\u00B7\u25AA\u25AB\u25CF\u25CB]\s+"
)
_NUMBERED_RE = re.compile(r"^(\d+)[.)]\s+(.+)", re.DOTALL)
_SEPARATOR_RE = re.compile(r"^[-_=*\u2014\u2015]{3,}\s*$")

# Code-like tokens that strongly suggest a code block
_CODE_TOKENS = re.compile(
    r"(\bdef\b|\bclass\b|\bimport\b|\bfrom\b.*\bimport\b|"
    r"\bfunction\b|\bconst\b|\blet\b|\bvar\b|\breturn\b|"
    r"#include|public static|void main|System\.out|"
    r"[{};]|=>|->|:=|\|\||&&|<<<|>>>|0x[0-9a-fA-F]+)"
)

# Short-line ratio threshold: if ≥ 70 % of lines are short, likely a list/code
_SHORT_LINE_RATIO = 0.7


def _clean_text(text: str) -> str:
    # Collapse internal whitespace; strip leading/trailing
    text = re.sub(r"[ \t]+", " ", text)
    text = "\n".join(line.strip() for line in text.splitlines())
    return text.strip()


def _is_heading(text: str, ratio: float, para: Paragraph) -> bool:
    lines = [l for l in text.splitlines() if l.strip()]
    if ratio < 1.3:
        return False
    # Headings are usually short (1–2 lines, few words)
    if len(lines) > 3:
        return False
    total_words = sum(len(l.split()) for l in lines)
    if total_words > 20:
        return False
    # Headings typically don't end with a period (unless abbreviation)
    last_line = lines[-1].rstrip()
    if last_line.endswith(".") and not last_line.endswith("..."):
        return False
    return True


def _is_code(text: str, para: Paragraph) -> bool:
    lines = [l for l in text.splitlines() if l.strip()]
    if not lines:
        return False
    # Strong code token match
    if _CODE_TOKENS.search(text):
        # Check that most lines look code-like (indented or short keywords)
        code_lines = sum(1 for l in lines if _CODE_TOKENS.search(l) or l.startswith("  "))
        if code_lines / len(lines) >= 0.4:
            return True
    # Indentation-based: all lines start with spaces/tabs (≥ 2 lines)
    if len(lines) >= 2:
        indented = sum(1 for l in lines if l.startswith("  ") or l.startswith("\t"))
        if indented / len(lines) >= 0.8:
            return True
    return False


def _is_bullet_list(text: str) -> bool:
    first_line = text.splitlines()[0] if text else ""
    return bool(_BULLET_RE.match(first_line))


def _is_numbered_list(text: str) -> bool:
    first_line = text.splitlines()[0] if text else ""
    return bool(_NUMBERED_RE.match(first_line))


def _is_separator(text: str) -> bool:
    return bool(_SEPARATOR_RE.match(text))


def _is_table_row(text: str, para: Paragraph) -> bool:
    # Heuristic: multiple "columns" — if there are ≥2 words spread widely
    if not para.lines:
        return False
    for line in para.lines:
        if len(line.words) >= 2:
            span = line.right - line.left
            gap_threshold = span * 0.25
            # Check for large horizontal gaps between consecutive words
            for i in range(len(line.words) - 1):
                gap = line.words[i + 1].left - line.words[i].right
                if gap >= gap_threshold:
                    return True
    return False


def _is_caption(text: str, para: Paragraph) -> bool:
    lines = [l for l in text.splitlines() if l.strip()]
    words = text.split()
    # Captions: short, below average height
    return (
        len(lines) <= 2
        and len(words) <= 15
        and para.avg_line_height < 12
        and not _is_heading(text, 1.0, para)
    )


def _indent_level(para: Paragraph, ocr: OcrResult) -> int:
    """Estimate indentation level 0/1/2 based on left offset."""
    left_positions = sorted(
        set(p.left for p in ocr.paragraphs if p.left > 0)
    )
    if not left_positions:
        return 0
    min_left = left_positions[0]
    avg_char_w = ocr.global_avg_word_height * 0.55  # rough char width
    offset = max(0, para.left - min_left)
    return min(3, int(offset / max(avg_char_w * 2, 8)))


def _strip_bullet(text: str) -> str:
    lines = text.splitlines()
    lines[0] = _BULLET_RE.sub("", lines[0])
    return "\n".join(lines)


def _parse_numbered(text: str) -> tuple[Optional[int], str]:
    m = _NUMBERED_RE.match(text)
    if m:
        return int(m.group(1)), m.group(2).strip()
    return None, text
