"""
labz — Local LLM toolkit CLI

Subcommands:
    labz chat     Chat with a local Ollama model
    labz img2md   Convert images/screenshots to Markdown
    labz imagine  Generate images from text prompts
    labz history  View and manage chat history
    labz models   List available local models
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import click
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.syntax import Syntax
from rich.table import Table

console     = Console(stderr=True)
out_console = Console(highlight=False)

OLLAMA_URL_DEFAULT = "http://localhost:11434"


# ──────────────────────────────────────────────────────────────────────────────
# Root group
# ──────────────────────────────────────────────────────────────────────────────

@click.group(invoke_without_command=True, context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(package_name="labz")
@click.pass_context
def cli(ctx: click.Context) -> None:
    """labz — Local LLM toolkit. Run everything on your own machine."""
    if ctx.invoked_subcommand is None:
        _print_home()


def _print_home() -> None:
    c = Console()
    c.print()
    c.print(Panel.fit(
        "[bold cyan]labz[/]  [dim]v0.1.0[/]\n"
        "[white]Local LLM toolkit — runs entirely on your machine.[/]\n"
        "[dim]No API keys. No cloud. No token costs.[/]",
        border_style="cyan", padding=(0, 2),
    ))

    c.print("\n[bold yellow]COMMANDS[/]")
    tbl = Table(show_header=False, box=None, padding=(0, 2), show_edge=False)
    tbl.add_column("cmd",  style="cyan",  no_wrap=True, min_width=14)
    tbl.add_column("desc", style="white")
    tbl.add_row("chat",    "Chat with a local Ollama model (mistral, llama3.2, …)")
    tbl.add_row("img2md",  "Convert screenshots / images to Markdown — save AI tokens")
    tbl.add_row("imagine", "Generate images from text prompts (Stable Diffusion)")
    tbl.add_row("history", "View, search, and delete saved chat sessions")
    tbl.add_row("models",  "List all locally available models")
    c.print(tbl)

    c.print("\n[bold yellow]QUICK START[/]")
    ex = Table(show_header=False, box=None, padding=(0, 2), show_edge=False)
    ex.add_column("desc",  style="dim",   min_width=36)
    ex.add_column("cmd",   style="green")
    ex.add_row("# Start a chat",                        "labz chat")
    ex.add_row("# Convert a screenshot to Markdown",    "labz img2md screenshot.png")
    ex.add_row("# Ask a question about an image",       'labz img2md screenshot.png --ask "What errors are shown?"')
    ex.add_row("# Generate an image",                   'labz imagine "a sunset over mountains"')
    ex.add_row("# View chat history",                   "labz history")
    ex.add_row("# List all local models",               "labz models")
    c.print(ex)

    c.print("\n[bold yellow]OLLAMA SETUP[/]  [dim](needed for chat and img2md --backend ollama)[/]")
    c.print(
        "  [green]brew install ollama[/]\n"
        "  [green]ollama pull mistral[/]           [dim]# chat model[/]\n"
        "  [green]ollama pull llama3.2-vision[/]   [dim]# vision model for img2md[/]\n"
    )
    c.print("[dim]Run 'labz <command> --help' for full options on any command.[/]\n")


# ──────────────────────────────────────────────────────────────────────────────
# labz chat
# ──────────────────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--model",    "-m", default=None,  help="Ollama model to use (default: auto-selects best available).")
@click.option("--private",  "-p", is_flag=True,  help="Don't save this session to history.")
@click.option("--ollama-url",     default=OLLAMA_URL_DEFAULT, show_default=True)
def chat(model: Optional[str], private: bool, ollama_url: str) -> None:
    """Chat with a local Ollama model."""
    from .ollama_backend import (
        _best_chat_model,
        can_reach_ollama,
        chat_session,
        ensure_ollama_running,
        list_chat_models,
    )
    from .history import new_session

    ensure_ollama_running(ollama_url)
    models = list_chat_models(ollama_url)
    if not models:
        if can_reach_ollama(ollama_url):
            console.print(
                "[yellow]Ollama is running, but no chat models are installed.[/]\n"
                "Run: [bold]ollama pull mistral[/]"
            )
        else:
            console.print(
                "[yellow]Ollama is not running.[/]\n"
                "Run: [bold]ollama serve[/]"
            )
        sys.exit(1)

    chosen = model or _best_chat_model(ollama_url)
    session = new_session()
    privacy = "[dim red](private — not saved)[/]" if private else f"[dim](saved to ~/.labz/history)[/]"

    console.print(
        f"\n[bold cyan]Chat[/] — model: [dim]{chosen}[/]  {privacy}\n"
        "[dim]Type anything. Enter 'exit' or Ctrl+C to quit.[/]\n"
    )
    chat_session(None, model=chosen, base_url=ollama_url, session=None if private else session)
    console.print("\n[dim]Chat ended.[/]")
    if not private and session.message_count > 0:
        console.print(f"[dim]Session saved: {session.id}[/]")


# ──────────────────────────────────────────────────────────────────────────────
# labz img2md
# ──────────────────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("images", nargs=-1, type=click.Path(exists=False))
@click.option("--out",          "-o", type=click.Path(), default=None,  help="Output .md file (default: <image>.md).")
@click.option("--out-dir",      "-d", type=click.Path(), default=None,  help="Output directory for batch mode.")
@click.option("--backend",      "-b", type=click.Choice(["auto","ocr","ollama"], case_sensitive=False), default="auto", show_default=True)
@click.option("--lang",         "-l", default="eng",   show_default=True, help="Tesseract language code, e.g. eng+fra.")
@click.option("--model",        "-m", default=None,    help="Override Ollama vision model.")
@click.option("--ask",                default=None,    help='Ask a question about the image, e.g. --ask "What errors?"')
@click.option("--chat",               is_flag=True,   help="Start interactive chat about the image after converting.")
@click.option("--chat-model",         default=None,    help="Override chat model for --ask / --chat.")
@click.option("--private",      "-p", is_flag=True,   help="Don't save chat session to history.")
@click.option("--stdout",             is_flag=True,   help="Print Markdown to stdout instead of writing a file.")
@click.option("--info",         "-i", is_flag=True,   help="Show stats + preview without saving.")
@click.option("--no-preprocess",      is_flag=True,   help="Skip image cleaning (faster, less accurate).")
@click.option("--ollama-url",         default=OLLAMA_URL_DEFAULT, show_default=True)
def img2md(
    images, out, out_dir, backend, lang, model, ask, chat,
    chat_model, private, stdout, info, no_preprocess, ollama_url,
) -> None:
    """Convert screenshots and images to Markdown locally.

    Saves 80-95% of AI tokens vs uploading raw images.

    Examples:

      labz img2md screenshot.png

      labz img2md screenshot.png notes.md

      labz img2md screenshot.png --ask "What errors are shown?"

      labz img2md *.png --out-dir ./markdown/
    """
    from .converter import ImgToMd

    if not images:
        console.print("[yellow]No images provided. Usage: labz img2md <image.png>[/]")
        sys.exit(1)

    images = list(images)
    if not out and len(images) >= 2 and str(images[-1]).lower().endswith(".md"):
        out = images.pop()
    images = [_resolve_path(p) for p in images]

    converter = ImgToMd(
        backend=backend, lang=lang, skip_preprocess=no_preprocess,
        ollama_model=model, ollama_url=ollama_url,
    )

    for image_path in images:
        try:
            _img2md_one(
                converter, image_path=image_path, out=out, out_dir=out_dir,
                stdout=stdout, info=info, ask=ask, do_chat=chat,
                chat_model=chat_model, ollama_url=ollama_url, private=private,
            )
        except Exception as exc:
            console.print(f"[bold red]ERROR[/] {image_path}: {exc}")


def _img2md_one(converter, *, image_path, out, out_dir, stdout, info,
                ask, do_chat, chat_model, ollama_url, private):
    from .converter import ImgToMd

    with Progress(SpinnerColumn(), TextColumn("{task.description}"),
                  TimeElapsedColumn(), console=console, transient=True) as p:
        t = p.add_task(f"Converting {Path(image_path).name} …", total=None)
        result = converter.convert(image_path)
        p.update(t, completed=True)

    _print_img2md_stats(result)

    if stdout or info:
        out_path = None
    elif out:
        out_path = out
    elif out_dir:
        d = Path(out_dir); d.mkdir(parents=True, exist_ok=True)
        out_path = str(d / Path(image_path).with_suffix(".md").name)
    else:
        out_path = str(Path(image_path).with_suffix(".md"))

    if info:
        preview = result.markdown[:800] + ("…" if len(result.markdown) > 800 else "")
        console.print(Panel(Syntax(preview, "markdown", theme="monokai", word_wrap=True),
                            title="Markdown preview", border_style="blue"))
    elif stdout:
        out_console.print(result.markdown, end="")
    else:
        saved = result.save(out_path)
        console.print(f"[bold green]Saved[/] → {saved}")

    if ask:
        _do_ask(result.markdown, ask, chat_model=chat_model, ollama_url=ollama_url)
    if do_chat:
        _do_img_chat(result.markdown, image_path=image_path, chat_model=chat_model,
                     ollama_url=ollama_url, private=private)


# ──────────────────────────────────────────────────────────────────────────────
# labz imagine
# ──────────────────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("prompt", nargs=-1)
@click.option("--out",    "-o", default=None,    type=click.Path(), help="Output image path (default: generated_<ts>.png).")
@click.option("--model",  "-m", default=None,    help="HuggingFace model ID (default: stabilityai/sdxl-turbo).")
@click.option("--width",        default=512,     show_default=True, help="Image width in pixels.")
@click.option("--height",       default=512,     show_default=True, help="Image height in pixels.")
@click.option("--steps",        default=None,    type=int,  help="Inference steps (auto-set per model).")
@click.option("--seed",         default=None,    type=int,  help="Random seed for reproducible results.")
@click.option("--negative",     default=None,    help="Negative prompt — things to avoid.")
@click.option("--list-cached",  is_flag=True,    help="List locally cached models and exit.")
def imagine(
    prompt, out, model, width, height, steps, seed, negative, list_cached,
) -> None:
    """Generate an image from a text prompt using Stable Diffusion.

    Runs locally on Apple Silicon (MPS), NVIDIA, or CPU.
    First run downloads the model (~7 GB, cached permanently).

    Examples:

      labz imagine "a golden retriever on a beach at sunset"

      labz imagine "futuristic city skyline" --out city.png --seed 42

      labz imagine "portrait photo" --model stabilityai/stable-diffusion-xl-base-1.0
    """
    from .imagine_backend import generate_image, list_cached_models, DEFAULT_MODEL

    if list_cached:
        cached = list_cached_models()
        if not cached:
            console.print("[dim]No models cached yet. Run labz imagine to download one.[/]")
        else:
            console.print("[bold]Locally cached image generation models:[/]")
            for m in cached:
                console.print(f"  • {m}")
        return

    prompt_text = " ".join(prompt).strip()
    if not prompt_text:
        console.print("[yellow]No prompt provided. Usage: labz imagine <prompt>[/]")
        sys.exit(1)

    model_id = model or DEFAULT_MODEL
    console.print(f"\n[bold cyan]Generating image[/]")
    console.print(f"  Prompt : [white]{prompt_text}[/]")
    console.print(f"  Model  : [dim]{model_id}[/]")
    console.print(f"  Size   : [dim]{width}×{height}[/]\n")

    kwargs = dict(
        model_id=model_id, width=width, height=height,
        output_path=out,
    )
    if steps:    kwargs["steps"]           = steps
    if seed:     kwargs["seed"]            = seed
    if negative: kwargs["negative_prompt"] = negative

    try:
        saved = generate_image(prompt_text, **kwargs)
        console.print(f"\n[bold green]Saved[/] → {saved}")
    except ImportError as e:
        console.print(f"[red]{e}[/]")
        sys.exit(1)


# ──────────────────────────────────────────────────────────────────────────────
# labz history
# ──────────────────────────────────────────────────────────────────────────────

@cli.group(invoke_without_command=True)
@click.pass_context
def history(ctx: click.Context) -> None:
    """View and manage saved chat sessions.

    \b
    Examples:
      labz history              list all sessions
      labz history delete <ID>  delete one session
      labz history delete all   delete all sessions
      labz history show <ID>    print full transcript
    """
    if ctx.invoked_subcommand is None:
        _history_list()


@history.command(name="delete")
@click.argument("target")
def history_delete(target: str) -> None:
    """Delete a session by ID, or 'all' to wipe everything."""
    from .history import delete_session, delete_all_sessions, list_sessions
    if target.lower() == "all":
        n = len(list_sessions())
        if n == 0:
            console.print("[dim]No sessions to delete.[/]"); return
        console.print(f"[yellow]This will permanently delete {n} session(s).[/]")
        if input("Type 'yes' to confirm: ").strip().lower() != "yes":
            console.print("[dim]Cancelled.[/]"); return
        console.print(f"[green]Deleted {delete_all_sessions()} session(s).[/]")
    else:
        if delete_session(target):
            console.print(f"[green]Deleted {target}.[/]")
        else:
            console.print(f"[red]Not found: {target}[/]  — run 'labz history' for IDs")


@history.command(name="show")
@click.argument("session_id")
def history_show(session_id: str) -> None:
    """Print the full transcript of a session."""
    from .history import load_session
    try:
        s = load_session(session_id)
    except FileNotFoundError:
        console.print(f"[red]Session not found: {session_id}[/]"); return

    console.print(Panel(
        f"[bold]Session[/] {s.id}\n"
        f"[dim]{s.started_dt.strftime('%Y-%m-%d %H:%M')}  •  "
        f"{s.message_count} question(s)"
        + (f"  •  image: {Path(s.image_path).name}" if s.image_path else ""),
        border_style="cyan",
    ))
    for msg in s.messages:
        role_style = "bold green" if msg.role == "user" else "bold blue"
        label = "You" if msg.role == "user" else "Assistant"
        console.print(f"\n[{role_style}]{label}:[/] {msg.content}")


def _history_list() -> None:
    from .history import list_sessions
    sessions = list_sessions()
    if not sessions:
        console.print("[dim]No saved sessions. Start one with: labz chat[/]"); return
    tbl = Table(title="Chat history", box=box.SIMPLE)
    tbl.add_column("ID",              style="cyan",  no_wrap=True)
    tbl.add_column("Date",            style="white", no_wrap=True)
    tbl.add_column("Turns",           style="white", justify="right")
    tbl.add_column("Image",           style="dim",   no_wrap=True)
    tbl.add_column("First question",  style="dim")
    for s in sessions:
        img = Path(s.image_path).name if s.image_path else "—"
        tbl.add_row(s.id, s.started_dt.strftime("%Y-%m-%d %H:%M"),
                    str(s.message_count), img, s.first_question)
    console.print(tbl)
    console.print("[dim]labz history show <ID>    view transcript[/]")
    console.print("[dim]labz history delete <ID>  delete session[/]")


# ──────────────────────────────────────────────────────────────────────────────
# labz models
# ──────────────────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--ollama-url", default=OLLAMA_URL_DEFAULT)
def models(ollama_url: str) -> None:
    """List all locally available models (Ollama + cached image gen models)."""
    from .ollama_backend import can_reach_ollama, ensure_ollama_running, list_chat_models, list_vision_models
    from .imagine_backend import list_cached_models

    # Ollama models
    ensure_ollama_running(ollama_url)
    vision = set(list_vision_models(ollama_url))
    all_ollama = list_chat_models(ollama_url)

    if all_ollama:
        tbl = Table(title="Ollama models", box=box.SIMPLE)
        tbl.add_column("Model",    style="cyan")
        tbl.add_column("Type",     style="white")
        tbl.add_column("Use with", style="dim")
        for m in all_ollama:
            if m in vision:
                mtype = "[green]vision + chat[/]"
                use   = "labz img2md --backend ollama  •  labz chat"
            else:
                mtype = "chat"
                use   = "labz chat  •  labz img2md --ask / --chat"
            tbl.add_row(m, mtype, use)
        console.print(tbl)
    else:
        if can_reach_ollama(ollama_url):
            console.print("[yellow]Ollama is running, but no models are installed.[/]  Run: ollama pull mistral")
        else:
            console.print("[yellow]Ollama is not running.[/]  Run: ollama serve")

    # Image generation models
    cached = list_cached_models()
    if cached:
        console.print()
        tbl2 = Table(title="Image generation models (cached)", box=box.SIMPLE)
        tbl2.add_column("Model",   style="cyan")
        tbl2.add_column("Use with", style="dim")
        for m in cached:
            tbl2.add_row(m, "labz imagine")
        console.print(tbl2)
    else:
        console.print("\n[dim]No image generation models cached yet.[/]  "
                      "Run: labz imagine \"a prompt\" to auto-download")

    if not all_ollama and not cached:
        console.print("\n[bold yellow]Getting started:[/]")
        console.print("  brew install ollama")
        console.print("  ollama pull mistral           # chat")
        console.print("  ollama pull llama3.2-vision   # vision")
        console.print('  labz imagine "test prompt"  # downloads SD model')


# ──────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────────────────────────────────────

def _resolve_path(raw: str) -> str:
    """Handle macOS Screenshot filenames with U+202F narrow no-break space."""
    import glob as _glob
    p = Path(raw)
    if p.exists(): return str(p)
    candidate = Path(raw.replace(" ", "\u202f"))
    if candidate.exists(): return str(candidate)
    matches = _glob.glob(str(p.parent / p.name.replace(" ", "?")))
    return matches[0] if matches else raw


def _do_ask(markdown: str, question: str, *, chat_model: Optional[str], ollama_url: str) -> None:
    from .ollama_backend import (
        _best_chat_model,
        ask_about_markdown,
        can_reach_ollama,
        ensure_ollama_running,
        list_chat_models,
    )
    ensure_ollama_running(ollama_url)
    if not list_chat_models(ollama_url):
        if can_reach_ollama(ollama_url):
            console.print("[yellow]Ollama is running, but no chat models are installed. Run: ollama pull mistral[/]"); return
        console.print("[yellow]Ollama not running — cannot answer. Run: ollama serve[/]"); return
    model = chat_model or _best_chat_model(ollama_url)
    console.print(f"\n[bold cyan]Asking[/] [dim]{model}[/]: {question}\n")
    with Progress(SpinnerColumn(), TextColumn("Thinking…"), console=console, transient=True) as p:
        p.add_task("", total=None)
        answer = ask_about_markdown(markdown, question, model=model, base_url=ollama_url)
    console.print(Panel(answer, title="Answer", border_style="green", padding=(1, 2)))


def _do_img_chat(markdown: str, *, image_path: str, chat_model: Optional[str],
                 ollama_url: str, private: bool) -> None:
    from .ollama_backend import (
        _best_chat_model,
        can_reach_ollama,
        chat_session,
        ensure_ollama_running,
        list_chat_models,
    )
    from .history import new_session
    ensure_ollama_running(ollama_url)
    if not list_chat_models(ollama_url):
        if can_reach_ollama(ollama_url):
            console.print("[yellow]Ollama is running, but no chat models are installed. Run: ollama pull mistral[/]"); return
        console.print("[yellow]Ollama not running — cannot chat. Run: ollama serve[/]"); return
    model = chat_model or _best_chat_model(ollama_url)
    session = new_session(image_path=image_path, markdown=markdown)
    privacy = "[dim red](private)[/]" if private else "[dim](saved)[/]"
    console.print(
        f"\n[bold cyan]Chat[/] — [dim]{model}[/] {privacy}\n"
        "[dim]Image context loaded. Enter 'exit' or Ctrl+C to quit.[/]\n"
    )
    chat_session(markdown, model=model, base_url=ollama_url,
                 session=None if private else session)
    console.print("\n[dim]Chat ended.[/]")
    if not private and session.message_count > 0:
        console.print(f"[dim]Session saved: {session.id}[/]")


def _print_img2md_stats(result) -> None:
    tbl = Table(show_header=False, box=None, padding=(0, 1))
    tbl.add_column("Key",   style="bold cyan", no_wrap=True)
    tbl.add_column("Value", style="white")
    tbl.add_row("Backend", result.backend_used)
    tbl.add_row("Image",   f"{result.image_width}×{result.image_height}px")
    if result.profile:
        p = result.profile
        cls_color = {"text_heavy": "green", "mixed": "yellow",
                     "graphic_heavy": "red"}.get(p.classification, "white")
        tbl.add_row("Image type",
                    f"[{cls_color}]{p.classification}[/] [dim](colour {p.color_saturation:.2f}, "
                    f"OCR conf {p.ocr_confidence:.0f}%, {p.ocr_word_count} words)[/]")
        tbl.add_row("Routing", f"[dim]{p.reason}[/]")
    tbl.add_row("Elapsed",                f"{result.elapsed_seconds:.1f}s")
    tbl.add_row("Markdown tokens (est.)", f"~{result.markdown_tokens:,}")
    tbl.add_row("Image tokens (est.)",    f"~{result.image_tokens_approx:,}")
    ratio = result.compression_ratio
    color = "green" if ratio < 0.8 else "yellow" if ratio < 1.2 else "red"
    tbl.add_row("Compression",
                f"[{color}]{ratio:.2f}x[/] [dim](saves ~{result.tokens_saved:,} tokens)[/]")
    console.print(tbl)
