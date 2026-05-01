"""
Video generation backend using LTX-Video via the diffusers library.

Runs locally on Apple Silicon (MPS), NVIDIA (CUDA), or CPU.

Default model: Lightricks/LTX-Video
  - ~7 GB download (one-time, cached in ~/.cache/huggingface)
  - ~5-10 min per 5s clip on M3
  - No API key, no internet after first download

Requirements:
    pip install labz[video]
    brew install ffmpeg      # needed to write .mp4 files
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Optional

DEFAULT_VIDEO_MODEL = "Lightricks/LTX-Video"


def _get_device():
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def generate_video(
    prompt: str,
    *,
    model_id: str = DEFAULT_VIDEO_MODEL,
    negative_prompt: str = "worst quality, inconsistent motion, blurry, jittery, distorted",
    width: int = 704,
    height: int = 480,
    num_frames: int = 25,
    fps: int = 24,
    steps: int = 50,
    guidance: float = 3.0,
    seed: Optional[int] = None,
    output_path: Optional[str] = None,
) -> str:
    """
    Generate a short video from *prompt* and save it as an MP4.

    Parameters
    ----------
    prompt : str
        Text description of the video to generate.
    model_id : str
        HuggingFace model ID. Downloaded automatically on first use (~7 GB).
    negative_prompt : str
        Things to avoid in the video.
    width, height : int
        Frame dimensions. Must be multiples of 32.
    num_frames : int
        Number of frames. 25 frames at 24fps ≈ 1 second; 120 frames ≈ 5 seconds.
    fps : int
        Frames per second in the output MP4.
    steps : int
        Inference steps — more steps = better quality but slower.
    guidance : float
        Guidance scale. 3.0 is a good default for LTX-Video.
    seed : int | None
        Random seed for reproducibility.
    output_path : str | None
        Where to save the MP4. Defaults to ./generated_video_<timestamp>.mp4

    Returns
    -------
    str
        Path to the saved video file.

    Raises
    ------
    ImportError
        If diffusers / torch / imageio are not installed.
    """
    try:
        import torch
        from diffusers import LTXPipeline
        from diffusers.utils import export_to_video
    except ImportError:
        raise ImportError(
            "Video generation requires additional packages.\n"
            "Run: pipx install --force 'labz[video] @ git+https://github.com/jlrosssc/labz.git'\n"
            "Also requires ffmpeg: brew install ffmpeg"
        )

    try:
        import imageio  # noqa: F401
    except ImportError:
        raise ImportError(
            "imageio is required to save video files.\n"
            "Run: pipx inject labz imageio imageio-ffmpeg\n"
            "Also requires ffmpeg: brew install ffmpeg"
        )

    device = _get_device()
    # bfloat16 is not supported on MPS — use float32 on Apple Silicon
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

    from rich.console import Console
    c = Console(stderr=True)
    c.print(f"[dim]Loading model [cyan]{model_id}[/] on [cyan]{device}[/] …[/]")
    c.print("[dim](First run downloads ~7 GB — subsequent runs are instant)[/]")
    c.print("[dim](If you need to stop it, use Ctrl+C, not Ctrl+Z.)[/]")

    pipe = LTXPipeline.from_pretrained(model_id, torch_dtype=dtype)
    pipe = pipe.to(device)

    if device != "cuda":
        pipe.enable_attention_slicing()

    generator = None
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)

    duration_s = num_frames / fps
    c.print(f"[dim]Generating {num_frames} frames ({duration_s:.1f}s at {fps}fps, {steps} steps) …[/]")
    c.print("[dim]This takes several minutes on Apple Silicon — please wait.[/]")

    kwargs: dict = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "width": width,
        "height": height,
        "num_frames": num_frames,
        "num_inference_steps": steps,
        "guidance_scale": guidance,
    }
    if generator is not None:
        kwargs["generator"] = generator

    result = pipe(**kwargs)
    frames = result.frames[0]

    if output_path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = str(Path.cwd() / f"generated_video_{ts}.mp4")

    export_to_video(frames, output_path, fps=fps)
    return output_path


def list_cached_video_models() -> list[str]:
    """Return HuggingFace video model IDs that are already cached locally."""
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    if not cache_dir.exists():
        return []
    known = {"lightricks--ltx-video": "Lightricks/LTX-Video"}
    models = []
    for d in cache_dir.iterdir():
        if d.is_dir() and d.name.startswith("models--"):
            key = d.name[len("models--"):]
            if key in known:
                models.append(known[key])
    return sorted(models)
