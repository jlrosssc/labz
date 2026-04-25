"""
Image profiler — analyzes an image and recommends the best conversion backend.

Three signals are combined to classify images:

  1. color_saturation  — mean HSV saturation
                         low  → monochrome/grayscale → likely text document
                         high → colorful             → likely photo or graphic UI

  2. text_coverage     — fraction of pixels that look like ink after thresholding
                         moderate coverage → dense text
                         very low or very high → photo / blank

  3. ocr_confidence    — quick Tesseract pass on a small thumbnail
                         high confidence → text is clean and readable → OCR is fine
                         low confidence  → text is stylized/absent    → LLM needed

Classification:
  text_heavy   → OCR  (fast, ~2-4s, high accuracy)
  mixed        → OCR first; if confidence poor, escalate to Ollama
  graphic_heavy→ Ollama (slower but understands visual content)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from PIL import Image, ImageFilter


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class ImageProfile:
    # Raw signals (0–1 unless noted)
    color_saturation: float    # mean HSV saturation
    text_coverage: float       # ink-pixel fraction after threshold
    edge_regularity: float     # how "text-like" the edges are (0–1)
    ocr_confidence: float      # Tesseract avg word confidence (0–100)
    ocr_word_count: int        # words found in quick pass

    # Decision
    classification: str        # "text_heavy" | "mixed" | "graphic_heavy"
    recommended_backend: str   # "ocr" | "ollama"
    reason: str                # human-readable explanation

    @property
    def is_text_heavy(self) -> bool:
        return self.classification == "text_heavy"

    @property
    def is_graphic_heavy(self) -> bool:
        return self.classification == "graphic_heavy"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def profile_image(
    img: Image.Image,
    *,
    ollama_available: bool = False,
    ocr_confidence_threshold: float = 65.0,
    saturation_text_max: float = 0.18,
    saturation_graphic_min: float = 0.38,
) -> ImageProfile:
    """
    Analyse *img* and return an :class:`ImageProfile` with a backend recommendation.

    Parameters
    ----------
    img:
        PIL Image to analyse.
    ollama_available:
        Whether Ollama is running and has a vision model. If False, OCR is
        always recommended (with a quality warning for graphic images).
    ocr_confidence_threshold:
        Minimum Tesseract confidence (0–100) to consider OCR "reliable".
    saturation_text_max:
        Images with mean HSV saturation below this are treated as text-like.
    saturation_graphic_min:
        Images with mean HSV saturation above this are treated as graphic-heavy.
    """
    # Work on a thumbnail for speed (analysis doesn't need full res)
    thumb = _thumbnail(img, max_side=600)

    saturation = _mean_saturation(thumb)
    text_cov = _text_coverage(thumb)
    edge_reg = _edge_regularity(thumb)
    conf, word_count = _quick_ocr(thumb)

    classification, backend, reason = _classify(
        saturation=saturation,
        text_coverage=text_cov,
        edge_regularity=edge_reg,
        ocr_confidence=conf,
        ocr_word_count=word_count,
        ollama_available=ollama_available,
        ocr_confidence_threshold=ocr_confidence_threshold,
        saturation_text_max=saturation_text_max,
        saturation_graphic_min=saturation_graphic_min,
    )

    return ImageProfile(
        color_saturation=saturation,
        text_coverage=text_cov,
        edge_regularity=edge_reg,
        ocr_confidence=conf,
        ocr_word_count=word_count,
        classification=classification,
        recommended_backend=backend,
        reason=reason,
    )


# ---------------------------------------------------------------------------
# Signal extractors
# ---------------------------------------------------------------------------

def _thumbnail(img: Image.Image, max_side: int = 600) -> Image.Image:
    img = img.copy().convert("RGB")
    img.thumbnail((max_side, max_side), Image.LANCZOS)
    return img


def _mean_saturation(img: Image.Image) -> float:
    """Mean HSV saturation (0–1). Low = greyscale/text, high = colourful."""
    hsv = np.array(img.convert("HSV"), dtype=np.float32)
    # PIL HSV: H in [0,255], S in [0,255], V in [0,255]
    return float(hsv[:, :, 1].mean() / 255.0)


def _text_coverage(img: Image.Image) -> float:
    """
    Fraction of pixels that look like 'ink' (dark on light background).
    Text documents typically sit between 0.04 and 0.35.
    """
    gray = np.array(img.convert("L"), dtype=np.float32)
    # Simple Otsu-like threshold: pixels darker than mean - 0.5*std
    mean, std = gray.mean(), gray.std()
    ink = gray < (mean - 0.4 * std)
    return float(ink.sum() / ink.size)


def _edge_regularity(img: Image.Image) -> float:
    """
    Score 0–1 measuring how 'text-like' the edge structure is.

    Text edges are numerous, fine, and distributed roughly uniformly across
    horizontal bands. Photos have fewer but smoother edges.

    Method: compute row-wise edge counts and measure their coefficient of
    variation (CV). Low CV → edges distributed evenly → text-like.
    We invert so high score = more text-like.
    """
    gray = img.convert("L").filter(ImageFilter.FIND_EDGES)
    arr = np.array(gray, dtype=np.float32)
    # Binarise edges
    edge_mask = arr > arr.mean() + arr.std()
    row_counts = edge_mask.sum(axis=1).astype(np.float64)
    if row_counts.mean() < 1:
        return 0.0
    cv = row_counts.std() / (row_counts.mean() + 1e-6)
    # CV close to 0 → uniform (text); high CV → uneven (photo)
    regularity = float(np.clip(1.0 - cv / 3.0, 0.0, 1.0))
    return regularity


def _quick_ocr(img: Image.Image) -> Tuple[float, int]:
    """
    Run Tesseract on the thumbnail with minimal config. Returns (avg_conf, word_count).
    Falls back to (0, 0) if pytesseract is not working.
    """
    try:
        import pytesseract
        data = pytesseract.image_to_data(
            img,
            config="--psm 3 --oem 1",   # fast LSTM engine
            output_type=pytesseract.Output.DICT,
        )
        confs = [c for c in data["conf"] if isinstance(c, (int, float)) and c >= 0]
        words = [t for t, c in zip(data["text"], data["conf"])
                 if t and t.strip() and isinstance(c, (int, float)) and c > 30]
        avg_conf = float(np.mean(confs)) if confs else 0.0
        return avg_conf, len(words)
    except Exception:
        return 0.0, 0


# ---------------------------------------------------------------------------
# Classification logic
# ---------------------------------------------------------------------------

def _classify(
    *,
    saturation: float,
    text_coverage: float,
    edge_regularity: float,
    ocr_confidence: float,
    ocr_word_count: int,
    ollama_available: bool,
    ocr_confidence_threshold: float,
    saturation_text_max: float,
    saturation_graphic_min: float,
) -> Tuple[str, str, str]:
    """Return (classification, backend, reason)."""

    ocr_reliable = ocr_confidence >= ocr_confidence_threshold
    has_text = ocr_word_count >= 10

    # --- text_heavy ---
    # Low colour, decent text density, OCR reads it well
    if (
        saturation <= saturation_text_max
        and text_coverage >= 0.03
        and text_coverage <= 0.70
        and ocr_reliable
        and has_text
    ):
        return (
            "text_heavy",
            "ocr",
            f"Low colour saturation ({saturation:.2f}), "
            f"OCR confidence {ocr_confidence:.0f}%, "
            f"{ocr_word_count} words detected — OCR is fast and accurate",
        )

    # --- graphic_heavy ---
    # High colour OR very few words found OR OCR confidence very low
    if (
        saturation >= saturation_graphic_min
        or (ocr_word_count < 5 and text_coverage < 0.04)
        or ocr_confidence < 35
    ):
        if ollama_available:
            return (
                "graphic_heavy",
                "ollama",
                f"High colour saturation ({saturation:.2f}) or low OCR confidence "
                f"({ocr_confidence:.0f}%) — local vision LLM will extract more information",
            )
        else:
            return (
                "graphic_heavy",
                "ocr",
                f"Graphic-heavy image (saturation {saturation:.2f}, "
                f"OCR confidence {ocr_confidence:.0f}%) — "
                "Ollama not available, using OCR (install Ollama for better results)",
            )

    # --- mixed ---
    # Somewhere in between — try OCR; if Ollama available and OCR isn't confident, use it
    if ollama_available and not ocr_reliable:
        return (
            "mixed",
            "ollama",
            f"Mixed content — colour {saturation:.2f}, OCR confidence {ocr_confidence:.0f}% "
            f"(below threshold {ocr_confidence_threshold:.0f}%) — "
            "using local vision LLM for better accuracy",
        )

    return (
        "mixed",
        "ocr",
        f"Mixed content — colour {saturation:.2f}, OCR confidence {ocr_confidence:.0f}%, "
        f"{ocr_word_count} words — OCR sufficient"
        + (" (install Ollama for richer extraction)" if not ollama_available else ""),
    )
