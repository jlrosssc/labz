"""
Optional Ollama backend — uses llama3.2-vision for image-to-Markdown and
qwen2.5:7b for chat. Both run locally via the Ollama HTTP API.

Requirements:
    pip install labz[ollama]          # pulls in httpx
    ollama pull llama3.2-vision       # vision model
    ollama pull qwen2.5:7b            # chat model
    ollama serve                      # (usually auto-started by Ollama)
"""

from __future__ import annotations

import base64
import json
import subprocess
import time
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

try:
    import httpx
    _HTTPX_AVAILABLE = True
except ImportError:
    _HTTPX_AVAILABLE = False


_DEFAULT_URL = "http://localhost:11434"
_CHAT_MODEL   = "qwen2.5:7b"
_VISION_MODEL = "llama3.2-vision"
_VISION_MODELS = [_VISION_MODEL]

_SYSTEM_PROMPT = """\
You are an expert document converter. \
Convert the content of this image into clean, well-structured Markdown. \
Follow these rules:
- Use # ## ### for headings based on visual hierarchy
- Use ``` fenced code blocks for any code (add language hint when obvious)
- Use | tables | for tabular data
- Use - for bullet lists and 1. for numbered lists
- Preserve ALL text from the image verbatim — do not summarise or omit
- Do NOT add commentary, explanations, or preamble — only output the Markdown
- If the image contains a diagram or non-textual figure, describe it briefly in italics: *[Figure: description]*
"""


def is_available(base_url: str = _DEFAULT_URL) -> bool:
    """Return True if Ollama is running and has at least one vision model."""
    models = _fetch_models(base_url, timeout=2.0)
    if models is None:
        return False
    return any(any(vm in m for vm in _VISION_MODELS) for m in models)


def list_vision_models(base_url: str = _DEFAULT_URL) -> list[str]:
    """Return names of locally available vision models."""
    models = _fetch_models(base_url, timeout=3.0)
    if models is None:
        return []
    return [m for m in models if any(vm in m for vm in _VISION_MODELS)]


def convert_with_ollama(
    image_path: str,
    *,
    model: Optional[str] = None,
    base_url: str = _DEFAULT_URL,
    timeout: float = 300.0,
) -> str:
    """
    Send *image_path* to a local Ollama vision model and return Markdown.

    Parameters
    ----------
    image_path:
        Path to the image file.
    model:
        Ollama model name. Auto-detects the best available vision model if None.
    base_url:
        Ollama API base URL (default: http://localhost:11434).
    timeout:
        Request timeout in seconds.

    Returns
    -------
    str
        The Markdown text produced by the model.

    Raises
    ------
    RuntimeError
        If Ollama is not available or the request fails.
    ImportError
        If ``httpx`` is not installed.
    """
    if not _HTTPX_AVAILABLE:
        raise ImportError(
            "httpx is required for the Ollama backend. "
            "Install it with: pip install imgmd[ollama]"
        )

    # Resolve model
    if model is None:
        available = list_vision_models(base_url)
        if not available:
            raise RuntimeError(
                f"Vision model not found in Ollama. "
                f"Run: ollama pull {_VISION_MODEL}"
            )
        model = available[0]

    # Read and base64-encode image
    image_data = base64.b64encode(Path(image_path).read_bytes()).decode()

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": _SYSTEM_PROMPT + "\n\nConvert this image to Markdown:",
                "images": [image_data],
            }
        ],
        "stream": False,
        "options": {
            "temperature": 0.0,  # deterministic output
        },
    }

    try:
        r = httpx.post(
            f"{base_url}/api/chat",
            json=payload,
            timeout=timeout,
        )
        r.raise_for_status()
    except httpx.HTTPStatusError as e:
        raise RuntimeError(f"Ollama request failed: {e.response.status_code} {e.response.text}") from e
    except httpx.RequestError as e:
        raise RuntimeError(f"Could not connect to Ollama at {base_url}: {e}") from e

    data = r.json()
    text = data.get("message", {}).get("content", "")
    if not text:
        raise RuntimeError(f"Ollama returned empty response: {data}")

    # Strip any markdown code fence that the model may have wrapped the output in
    text = _unwrap_outer_fence(text)
    return text.strip() + "\n"


def list_chat_models(base_url: str = _DEFAULT_URL) -> list[str]:
    """Return all locally available Ollama models (vision + text)."""
    models = _fetch_models(base_url, timeout=3.0)
    if models is None:
        return []
    return models


def can_reach_ollama(base_url: str = _DEFAULT_URL) -> bool:
    """Return True if the Ollama HTTP API is reachable."""
    return _fetch_models(base_url, timeout=2.0) is not None


def ensure_ollama_running(base_url: str = _DEFAULT_URL, startup_timeout: float = 6.0) -> bool:
    """Try to start a local Ollama server when the API is not reachable."""
    if can_reach_ollama(base_url):
        return True
    if not _is_local_ollama_url(base_url):
        return False

    try:
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except (FileNotFoundError, OSError):
        return False

    deadline = time.time() + startup_timeout
    while time.time() < deadline:
        if can_reach_ollama(base_url):
            return True
        time.sleep(0.25)
    return False


def _best_chat_model(base_url: str) -> str:
    """Return qwen2.5:7b if installed, otherwise raise with install instructions."""
    available = list_chat_models(base_url)
    if not available:
        raise RuntimeError(
            f"No models found in Ollama. Run: ollama pull {_CHAT_MODEL}"
        )
    for m in available:
        if _CHAT_MODEL in m.lower():
            return m
    raise RuntimeError(
        f"Chat model {_CHAT_MODEL!r} not installed. Run: ollama pull {_CHAT_MODEL}"
    )


def ask_about_markdown(
    markdown: str,
    question: str,
    *,
    model: Optional[str] = None,
    base_url: str = _DEFAULT_URL,
    timeout: float = 300.0,
) -> str:
    """
    Ask a single question about image content that has already been converted
    to Markdown. Uses a local text/chat model — no image upload needed.

    Parameters
    ----------
    markdown : str
        The Markdown representation of the image.
    question : str
        The question to ask about the content.
    model : str | None
        Ollama model name. Auto-selects best available if None.
    base_url : str
        Ollama API base URL.
    timeout : float
        Request timeout in seconds.

    Returns
    -------
    str
        The model's answer.
    """
    if not _HTTPX_AVAILABLE:
        raise ImportError("httpx is required. Install with: pip install imgmd[ollama]")

    if model is None:
        model = _best_chat_model(base_url)

    system = (
        "You are a helpful assistant. The user has provided the text content "
        "of an image, already converted to Markdown. Answer questions about it "
        "concisely and accurately. Refer only to what is in the content."
    )

    messages = [
        {
            "role": "user",
            "content": (
                f"Here is the content of an image as Markdown:\n\n"
                f"```markdown\n{markdown.strip()}\n```\n\n"
                f"Question: {question}"
            ),
        }
    ]

    return _chat(messages, system=system, model=model, base_url=base_url, timeout=timeout)


def chat_session(
    markdown: Optional[str],
    *,
    model: Optional[str] = None,
    base_url: str = _DEFAULT_URL,
    timeout: float = 300.0,
    session=None,        # imgmd.history.ChatSession | None
) -> None:
    """
    Start an interactive multi-turn chat session.

    If markdown is provided it is used as context (image mode).
    If markdown is None, opens a plain general-purpose chat.
    If session is provided, messages are appended to it and saved after each turn.
    Reads from stdin. Type 'exit' or 'quit' or press Ctrl+C to end.
    """
    if not _HTTPX_AVAILABLE:
        raise ImportError("httpx is required. Install with: pip install imgmd[ollama]")

    if model is None:
        model = _best_chat_model(base_url)

    if markdown:
        system = (
            "You are a helpful assistant. The user has provided the text content "
            "of an image, already converted to Markdown. Answer questions about it "
            "concisely and accurately. Refer only to what is in the content."
        )
        # Seed context — not stored in persistent history (it's the image itself)
        seed: list[dict] = [
            {
                "role": "user",
                "content": (
                    f"Here is the content of an image as Markdown:\n\n"
                    f"```markdown\n{markdown.strip()}\n```\n\n"
                    f"I'll ask you questions about it."
                ),
            },
            {
                "role": "assistant",
                "content": "Understood. What would you like to know about this image?",
            },
        ]
    else:
        system = "You are a helpful, concise assistant."
        seed = []

    # If resuming a session, replay its saved messages as context
    prior: list[dict] = session.to_ollama_history() if session else []
    history: list[dict] = seed + prior

    try:
        while True:
            try:
                question = input("\nYou: ").strip()
            except EOFError:
                break

            if not question:
                continue
            if question.lower() in ("exit", "quit", "bye", "q"):
                break

            history.append({"role": "user", "content": question})
            answer = _chat(history, system=system, model=model, base_url=base_url, timeout=timeout)
            history.append({"role": "assistant", "content": answer})
            print(f"\nAssistant: {answer}")

            # Persist each turn immediately so nothing is lost on Ctrl+C
            if session is not None:
                session.add("user", question)
                session.add("assistant", answer)
                from .history import save_session
                save_session(session)

    except KeyboardInterrupt:
        pass


def _chat(
    messages: list[dict],
    *,
    system: str,
    model: str,
    base_url: str,
    timeout: float,
) -> str:
    """Low-level chat call to Ollama /api/chat."""
    payload = {
        "model": model,
        "system": system,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.3},
    }
    try:
        r = httpx.post(f"{base_url}/api/chat", json=payload, timeout=timeout)
        r.raise_for_status()
    except httpx.HTTPStatusError as e:
        raise RuntimeError(f"Ollama error: {e.response.status_code} {e.response.text}") from e
    except httpx.RequestError as e:
        raise RuntimeError(f"Could not connect to Ollama at {base_url}: {e}") from e

    text = r.json().get("message", {}).get("content", "").strip()
    if not text:
        raise RuntimeError("Ollama returned an empty response")
    return text


def _unwrap_outer_fence(text: str) -> str:
    """If the model wrapped the entire output in ```markdown ... ```, strip it."""
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1])
    return text


def _fetch_models(base_url: str, timeout: float) -> Optional[list[str]]:
    """Return model names, [] when reachable but empty, or None when unreachable."""
    if not _HTTPX_AVAILABLE:
        return None
    try:
        r = httpx.get(f"{base_url}/api/tags", timeout=timeout)
        if r.status_code != 200:
            return None
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return None


def _is_local_ollama_url(base_url: str) -> bool:
    host = (urlparse(base_url).hostname or "").lower()
    return host in {"localhost", "127.0.0.1"}
