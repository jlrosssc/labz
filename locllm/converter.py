"""
Main converter — the public-facing ImgToMd class.

Backends
--------
  "ocr"     Pure Tesseract OCR + heuristic layout analysis.
            Works 100% offline, no extra downloads required.
            Great for: text-heavy screenshots, terminal output, code editors.

  "ollama"  Local vision LLM via Ollama.
            Much better for complex layouts, tables, diagrams, and UI screenshots.
            Requires: `ollama pull llava` (or llama3.2-vision, moondream, etc.)

  "auto"    Try Ollama first; fall back to OCR if Ollama is not available.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from PIL import Image

from .ocr import ocr_image
from .structure import classify
from .markdown import render, estimate_tokens, image_token_cost
from .analyze import profile_image, ImageProfile


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ConversionResult:
    markdown: str
    backend_used: str                  # "ocr" or "ollama"
    image_path: str
    image_width: int
    image_height: int
    elapsed_seconds: float

    # Token estimates
    markdown_tokens: int = 0
    image_tokens_approx: int = 0

    # Smart routing info (only set when backend="auto")
    profile: Optional[ImageProfile] = None

    @property
    def tokens_saved(self) -> int:
        """Rough number of tokens saved vs. sending the raw image."""
        return max(0, self.image_tokens_approx - self.markdown_tokens)

    @property
    def compression_ratio(self) -> float:
        if self.image_tokens_approx == 0:
            return 1.0
        return self.markdown_tokens / self.image_tokens_approx

    def __str__(self) -> str:
        return self.markdown

    def save(self, path: str | None = None) -> str:
        """Write markdown to *path* (default: same name as image but .md)."""
        if path is None:
            path = str(Path(self.image_path).with_suffix(".md"))
        Path(path).write_text(self.markdown, encoding="utf-8")
        return path


# ---------------------------------------------------------------------------
# Converter
# ---------------------------------------------------------------------------

class ImgToMd:
    """
    Convert images / screenshots to Markdown locally.

    Parameters
    ----------
    backend : str
        ``"ocr"`` (default), ``"ollama"``, or ``"auto"``.
    lang : str
        Tesseract language code. Default ``"eng"``.
        Pass ``"eng+fra"`` for mixed English/French, etc.
    skip_preprocess : bool
        Skip image cleaning (faster but less accurate for OCR backend).
    ollama_model : str | None
        Override the Ollama model name (e.g. ``"llava:13b"``).
    ollama_url : str
        Base URL for Ollama API. Default ``"http://localhost:11434"``.
    detect_code_language : bool
        Whether to auto-detect fenced code block language. Default True.
    """

    def __init__(
        self,
        backend: str = "auto",
        *,
        lang: str = "eng",
        skip_preprocess: bool = False,
        ollama_model: Optional[str] = None,
        ollama_url: str = "http://localhost:11434",
        detect_code_language: bool = True,
    ) -> None:
        if backend not in ("ocr", "ollama", "auto"):
            raise ValueError(f"backend must be 'ocr', 'ollama', or 'auto', got {backend!r}")
        self.backend = backend
        self.lang = lang
        self.skip_preprocess = skip_preprocess
        self.ollama_model = ollama_model
        self.ollama_url = ollama_url
        self.detect_code_language = detect_code_language

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def convert(self, image_path: str) -> ConversionResult:
        """
        Convert a single image to Markdown.

        Parameters
        ----------
        image_path : str
            Path to the image (PNG, JPEG, WebP, BMP, TIFF, GIF, …).

        Returns
        -------
        ConversionResult
        """
        path = str(Path(image_path).resolve())
        img = Image.open(path)
        w, h = img.size

        t0 = time.perf_counter()
        backend_used, markdown, prof = self._run(path, img)
        elapsed = time.perf_counter() - t0

        md_tokens = estimate_tokens(markdown)
        img_tokens = image_token_cost(w, h)

        return ConversionResult(
            markdown=markdown,
            backend_used=backend_used,
            image_path=path,
            image_width=w,
            image_height=h,
            elapsed_seconds=elapsed,
            markdown_tokens=md_tokens,
            image_tokens_approx=img_tokens,
            profile=prof,
        )

    def convert_batch(self, image_paths: List[str]) -> List[ConversionResult]:
        """Convert multiple images. Returns results in the same order."""
        return [self.convert(p) for p in image_paths]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run(self, path: str, img: Image.Image) -> tuple[str, str, Optional[ImageProfile]]:
        backend = self.backend
        prof: Optional[ImageProfile] = None

        if backend == "auto":
            from .ollama_backend import is_available
            ollama_up = is_available(self.ollama_url)

            # Profile the image to pick the best backend
            prof = profile_image(img, ollama_available=ollama_up)
            backend = prof.recommended_backend

        if backend == "ollama":
            md, be = self._run_ollama(path)
            return be, md, prof
        else:
            md, be = self._run_ocr(path, img)

            # If auto-routed to OCR but OCR confidence was poor, escalate to Ollama
            if (
                prof is not None
                and prof.classification == "mixed"
                and prof.ocr_confidence < 55
            ):
                from .ollama_backend import is_available
                if is_available(self.ollama_url):
                    md, be = self._run_ollama(path)

            return be, md, prof

    def _run_ocr(self, path: str, img: Image.Image) -> tuple[str, str]:
        ocr_result = ocr_image(
            img,
            lang=self.lang,
            skip_preprocess=self.skip_preprocess,
        )
        blocks = classify(ocr_result)
        markdown = render(blocks, detect_language=self.detect_code_language)
        return markdown, "ocr"

    def _run_ollama(self, path: str) -> tuple[str, str]:
        from .ollama_backend import convert_with_ollama
        markdown = convert_with_ollama(
            path,
            model=self.ollama_model,
            base_url=self.ollama_url,
        )
        return markdown, "ollama"
