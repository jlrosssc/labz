"""
Image preprocessing to improve OCR accuracy.

Steps applied:
  1. Upscale small images (< 1200px wide) to help Tesseract
  2. Convert to grayscale
  3. Deskew (correct rotation < 15°)
  4. Adaptive thresholding (handles uneven lighting / shadows)
  5. Light denoising
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
from PIL import Image, ImageFilter, ImageOps


# ---------------------------------------------------------------------------
# Public
# ---------------------------------------------------------------------------

def preprocess(img: Image.Image, *, target_dpi: int = 300) -> Image.Image:
    """Return a cleaned copy of *img* optimised for Tesseract."""
    img = img.convert("RGB")
    img = _upscale(img, min_width=1200)
    gray = _to_gray(img)
    gray = _deskew(gray)
    gray = _binarize(gray)
    gray = _denoise(gray)
    return gray


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _upscale(img: Image.Image, min_width: int = 1200) -> Image.Image:
    w, h = img.size
    if w < min_width:
        scale = min_width / w
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img


def _to_gray(img: Image.Image) -> Image.Image:
    return ImageOps.grayscale(img)


def _deskew(gray: Image.Image) -> Image.Image:
    """Rotate image to fix slight tilt using projection-profile method."""
    try:
        angle = _detect_skew(gray)
        if abs(angle) > 0.3:
            gray = gray.rotate(angle, expand=True, fillcolor=255)
    except Exception:
        pass  # skip deskew if it fails; better to OCR a tilted image than crash
    return gray


def _detect_skew(gray: Image.Image, max_angle: float = 15.0) -> float:
    """Return the estimated skew angle in degrees (positive = clockwise)."""
    arr = np.array(gray, dtype=np.uint8)
    # Binarise quickly
    thresh = arr < 128
    angles = np.linspace(-max_angle, max_angle, num=61)
    best_angle = 0.0
    best_score = -1.0

    for angle in angles:
        rotated = _np_rotate(thresh.astype(np.float32), angle)
        # Score = variance of row sums (high variance → text lines are horizontal)
        row_sums = rotated.sum(axis=1)
        score = float(np.var(row_sums))
        if score > best_score:
            best_score = score
            best_angle = angle

    return best_angle


def _np_rotate(arr: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate a 2-D float array by *angle_deg* using PIL (fast, good enough)."""
    img = Image.fromarray((arr * 255).astype(np.uint8))
    img = img.rotate(angle_deg, expand=False, fillcolor=0)
    return np.array(img).astype(np.float32) / 255.0


def _binarize(gray: Image.Image) -> Image.Image:
    """Adaptive thresholding: converts to pure B&W, handles shadows."""
    arr = np.array(gray, dtype=np.uint8)
    # Block-based local mean threshold (similar to cv2 ADAPTIVE_THRESH_MEAN)
    block = 31
    pad = block // 2
    padded = np.pad(arr, pad, mode="reflect")
    h, w = arr.shape
    # Use uniform filter via repeated sliding window (approx, fast)
    from PIL import ImageFilter as IF
    blurred = gray.filter(IF.BoxBlur(pad))
    local_mean = np.array(blurred, dtype=np.int16)
    binary = ((arr.astype(np.int16) > local_mean - 10)).astype(np.uint8) * 255
    return Image.fromarray(binary)


def _denoise(gray: Image.Image) -> Image.Image:
    """Very light median-style denoise to remove salt-and-pepper noise."""
    return gray.filter(ImageFilter.MedianFilter(size=3))
