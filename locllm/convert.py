"""
Convenience one-liner: ``imgmd.convert(path)``
"""

from __future__ import annotations
from .converter import ImgToMd


def convert(
    image_path: str,
    *,
    backend: str = "auto",
    lang: str = "eng",
) -> str:
    """
    Convert *image_path* to Markdown and return the string.

    This is a shortcut for::

        ImgToMd(backend=backend, lang=lang).convert(image_path).markdown

    Parameters
    ----------
    image_path : str
        Path to the image file.
    backend : str
        ``"auto"`` (default), ``"ocr"``, or ``"ollama"``.
    lang : str
        Tesseract language code. Default ``"eng"``.
    """
    return ImgToMd(backend=backend, lang=lang).convert(image_path).markdown
