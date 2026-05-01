# labz

Local LLM toolkit for:

- chat with local Ollama models
- image-to-Markdown conversion with `img2md`-style flows
- local image generation

## Fedora Install

Tested packaging target for this repo is Python 3.11.

### 1. Install system packages

```bash
sudo dnf install -y git python3.11 python3.11-pip pipx tesseract tesseract-langpack-eng
pipx ensurepath
```

Open a new shell after `pipx ensurepath`, or source your shell profile.

### 2. Install Ollama

Official Linux install command from Ollama:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Then start it:

```bash
ollama serve
```

In another terminal, pull the models you want:

```bash
ollama pull qwen2.5:7b        # chat model
ollama pull llama3.2-vision   # vision model for img2md
```

### 3. Install `labz`

For chat + image-to-Markdown:

```bash
pipx install --python /usr/bin/python3.11 'git+https://github.com/jlrosssc/labz.git[ollama]'
```

For chat + image-to-Markdown + image generation:

```bash
pipx install --python /usr/bin/python3.11 'git+https://github.com/jlrosssc/labz.git[all]'
```

### 4. Basic usage

```bash
labz chat
labz img2md screenshot.png
labz img2md screenshot.png --chat
labz imagine create a png of a leaf
```

## Notes

- If you stop a long model download, use `Ctrl+C`, not `Ctrl+Z`.
- The first image-generation run downloads model weights and can take a while.
- On Apple Silicon, `labz` uses safer full-precision image generation to avoid black images.
