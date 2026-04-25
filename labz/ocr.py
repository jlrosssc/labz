"""
OCR extraction layer — wraps Tesseract and returns structured word-level data.

Each word comes back with:
  - text, confidence
  - bounding box (left, top, width, height)
  - block / paragraph / line / word numbers (Tesseract hierarchy)

Higher-level callers (structure.py) group these into logical blocks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

import pytesseract
from PIL import Image

from .preprocess import preprocess


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Word:
    text: str
    conf: float          # 0‒100
    left: int
    top: int
    width: int
    height: int
    block_num: int
    par_num: int
    line_num: int
    word_num: int

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height

    @property
    def area(self) -> int:
        return self.width * self.height


@dataclass
class Line:
    words: List[Word] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words if w.text.strip())

    @property
    def top(self) -> int:
        return min(w.top for w in self.words) if self.words else 0

    @property
    def left(self) -> int:
        return min(w.left for w in self.words) if self.words else 0

    @property
    def bottom(self) -> int:
        return max(w.bottom for w in self.words) if self.words else 0

    @property
    def right(self) -> int:
        return max(w.right for w in self.words) if self.words else 0

    @property
    def height(self) -> int:
        return self.bottom - self.top

    @property
    def avg_word_height(self) -> float:
        if not self.words:
            return 0.0
        return sum(w.height for w in self.words) / len(self.words)

    @property
    def avg_conf(self) -> float:
        if not self.words:
            return 0.0
        return sum(w.conf for w in self.words) / len(self.words)


@dataclass
class Paragraph:
    lines: List[Line] = field(default_factory=list)
    block_num: int = 0
    par_num: int = 0

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines)

    @property
    def top(self) -> int:
        return min(l.top for l in self.lines) if self.lines else 0

    @property
    def left(self) -> int:
        return min(l.left for l in self.lines) if self.lines else 0

    @property
    def avg_line_height(self) -> float:
        if not self.lines:
            return 0.0
        return sum(l.avg_word_height for l in self.lines) / len(self.lines)

    @property
    def all_words(self) -> List[Word]:
        return [w for l in self.lines for w in l.words]


@dataclass
class OcrResult:
    paragraphs: List[Paragraph]
    image_width: int
    image_height: int

    @property
    def all_words(self) -> List[Word]:
        return [w for p in self.paragraphs for w in p.all_words]

    @property
    def global_avg_word_height(self) -> float:
        words = [w for w in self.all_words if w.height > 0 and w.conf > 40]
        if not words:
            return 12.0
        return sum(w.height for w in words) / len(words)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ocr_image(
    source: str | Image.Image,
    *,
    lang: str = "eng",
    skip_preprocess: bool = False,
    min_conf: float = 30.0,
) -> OcrResult:
    """
    Run Tesseract on *source* and return a structured :class:`OcrResult`.

    Parameters
    ----------
    source:
        File path or PIL Image.
    lang:
        Tesseract language code (default ``"eng"``).
        Use ``"eng+fra"`` for multi-language, etc.
    skip_preprocess:
        If True, skip image cleaning step (faster, less accurate).
    min_conf:
        Discard words with confidence below this value (0‒100).
    """
    if isinstance(source, str):
        img = Image.open(source)
    else:
        img = source.copy()

    orig_w, orig_h = img.size

    if not skip_preprocess:
        img = preprocess(img)

    proc_w, proc_h = img.size

    raw = pytesseract.image_to_data(
        img,
        lang=lang,
        output_type=pytesseract.Output.DICT,
        config="--psm 3",   # fully automatic page segmentation
    )

    # Scale factor (preprocessing may have upscaled)
    sx = orig_w / proc_w
    sy = orig_h / proc_h

    words = _parse_raw(raw, min_conf=min_conf, sx=sx, sy=sy)
    paragraphs = _group_into_paragraphs(words)
    return OcrResult(paragraphs=paragraphs, image_width=orig_w, image_height=orig_h)


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

def _parse_raw(
    raw: dict,
    *,
    min_conf: float,
    sx: float,
    sy: float,
) -> List[Word]:
    words: List[Word] = []
    n = len(raw["text"])
    for i in range(n):
        text = raw["text"][i]
        if not text or not text.strip():
            continue
        conf = float(raw["conf"][i])
        if conf < min_conf:
            continue
        words.append(Word(
            text=text.strip(),
            conf=conf,
            left=int(raw["left"][i] * sx),
            top=int(raw["top"][i] * sy),
            width=int(raw["width"][i] * sx),
            height=int(raw["height"][i] * sy),
            block_num=raw["block_num"][i],
            par_num=raw["par_num"][i],
            line_num=raw["line_num"][i],
            word_num=raw["word_num"][i],
        ))
    return words


def _group_into_paragraphs(words: List[Word]) -> List[Paragraph]:
    """Group words by (block_num, par_num) → (line_num) using Tesseract IDs."""
    # Map: (block, par) → {line → [words]}
    structure: dict[tuple, dict[int, list]] = {}
    for w in words:
        key = (w.block_num, w.par_num)
        if key not in structure:
            structure[key] = {}
        if w.line_num not in structure[key]:
            structure[key][w.line_num] = []
        structure[key][w.line_num].append(w)

    paragraphs: List[Paragraph] = []
    for (blk, par), lines_dict in sorted(structure.items()):
        lines = []
        for line_num in sorted(lines_dict):
            line_words = sorted(lines_dict[line_num], key=lambda w: w.left)
            lines.append(Line(words=line_words))
        p = Paragraph(lines=lines, block_num=blk, par_num=par)
        if p.text.strip():
            paragraphs.append(p)

    # Sort top-to-bottom, left-to-right
    paragraphs.sort(key=lambda p: (p.top, p.left))
    return paragraphs
