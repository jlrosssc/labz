"""
Image generation backend using Stable Diffusion via the diffusers library.

Runs entirely locally on Apple Silicon (MPS), NVIDIA (CUDA), or CPU.

Default model: stabilityai/sdxl-turbo
  - ~6.7 GB download (one-time, cached in ~/.cache/huggingface)
  - ~10–20s per image on M3 Pro
  - No API key, no internet after first download

Other supported models (pass via --model):
  stabilityai/stable-diffusion-xl-base-1.0   best quality, slower
  runwayml/stable-diffusion-v1-5             classic, 4 GB, fast
  stabilityai/stable-diffusion-2-1           good balance
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

DEFAULT_MODEL = "stabilityai/sdxl-turbo"

# Steps and guidance tuned per model type
_MODEL_DEFAULTS: dict[str, dict] = {
    "sdxl-turbo":          {"steps": 4,  "guidance": 0.0},
    "stable-diffusion-xl": {"steps": 30, "guidance": 7.5},
    "stable-diffusion-2":  {"steps": 30, "guidance": 7.5},
    "stable-diffusion-v1": {"steps": 30, "guidance": 7.5},
}


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


def _model_params(model_id: str) -> dict:
    for key, params in _MODEL_DEFAULTS.items():
        if key in model_id:
            return params
    return {"steps": 30, "guidance": 7.5}


def generate_image(
    prompt: str,
    *,
    model_id: str = DEFAULT_MODEL,
    negative_prompt: str = "blurry, low quality, distorted, watermark",
    width: int = 512,
    height: int = 512,
    steps: Optional[int] = None,
    guidance: Optional[float] = None,
    seed: Optional[int] = None,
    output_path: Optional[str] = None,
) -> str:
    """
    Generate an image from *prompt* and save it locally.

    Parameters
    ----------
    prompt : str
        Text description of the image to generate.
    model_id : str
        HuggingFace model ID. Downloaded automatically on first use.
    negative_prompt : str
        Things to avoid in the image.
    width, height : int
        Output dimensions. Must be multiples of 8.
    steps : int | None
        Inference steps. Auto-set per model if None.
    guidance : float | None
        Classifier-free guidance scale. Auto-set per model if None.
    seed : int | None
        Random seed for reproducibility.
    output_path : str | None
        Where to save the image. Defaults to ./generated_<timestamp>.png

    Returns
    -------
    str
        Path to the saved image file.

    Raises
    ------
    ImportError
        If diffusers / torch are not installed.
    """
    try:
        import torch
        from diffusers import AutoPipelineForText2Image
    except ImportError:
        raise ImportError(
            "Image generation requires additional packages.\n"
            "Install with: pip install labz[imagine]\n"
            "  or: pip install diffusers transformers accelerate torch"
        )

    device = _get_device()
    params = _model_params(model_id)
    steps   = steps   if steps   is not None else params["steps"]
    guidance = guidance if guidance is not None else params["guidance"]

    dtype = torch.float16 if device in ("mps", "cuda") else torch.float32

    from rich.console import Console
    c = Console(stderr=True)
    c.print(f"[dim]Loading model [cyan]{model_id}[/] on [cyan]{device}[/] …[/]")
    c.print("[dim](First run downloads the model — subsequent runs are instant)[/]")

    pipe = AutoPipelineForText2Image.from_pretrained(
        model_id,
        torch_dtype=dtype,
        use_safetensors=True,
        variant="fp16" if dtype == torch.float16 else None,
    )
    pipe = pipe.to(device)

    # Memory optimization for MPS / CPU
    if device != "cuda":
        pipe.enable_attention_slicing()

    generator = None
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)

    c.print(f"[dim]Generating ({steps} steps) …[/]")

    kwargs: dict = {
        "prompt": prompt,
        "width": width,
        "height": height,
        "num_inference_steps": steps,
    }
    if guidance > 0:
        kwargs["guidance_scale"] = guidance
        kwargs["negative_prompt"] = negative_prompt
    if generator:
        kwargs["generator"] = generator

    result = pipe(**kwargs)
    image = result.images[0]

    if output_path is None:
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = str(Path.cwd() / f"generated_{ts}.png")

    image.save(output_path)
    return output_path


def list_cached_models() -> list[str]:
    """Return HuggingFace model IDs that are already cached locally."""
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    if not cache_dir.exists():
        return []
    models = []
    for d in cache_dir.iterdir():
        if d.is_dir() and d.name.startswith("models--"):
            # Convert "models--stabilityai--sdxl-turbo" → "stabilityai/sdxl-turbo"
            parts = d.name[len("models--"):].split("--", 1)
            if len(parts) == 2:
                models.append(f"{parts[0]}/{parts[1]}")
    return sorted(models)
