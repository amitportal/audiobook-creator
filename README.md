# Audiobook Creator

**Version 1.1.0** - Convert Markdown books into high-quality MP3 audiobooks using multiple TTS models (Supertonic, Kokoro, Chatterbox).

## Features

- 🎙️ **Modular TTS Architecture**: Support for multiple TTS backends:
  - **Supertone Supertonic** (Default): Ultra-fast ONNX-based models.
  - **Kokoro**: Lightweight, high-quality open-weight model.
  - **ResembleAI Chatterbox**: Expressive model with paralinguistic tag support.
- 📚 **Smart Chapter Detection**: Automatically detects chapters from Markdown headings (including generic H1).
- 🎵 **MP3 Output**: Professional MP3 audiobooks with automatic conversion.
- 🛠️ **Automatic FFmpeg Detection**: Robust search for FFmpeg on Windows (WinGet, Program Files, etc.).
- ⚡ **Performance Optimized**: In-memory audio processing (no intermediate WAV files).
- 💾 **Smart Caching**: Skip re-generating chapters that already have an output file with `--use-cache`.
- 🧠 **Dynamic Pauses**: Semantic similarity-based pauses between text chunks for natural flow.
- 📢 **Clear Progress**: Real-time console output "yelling" the current chapter being processed.
- 🎚️ **Configurable Voice Styles**: Choose different voice styles depending on the model.

## Quick Start

### Installation

```bash
# Prerequisites: Python 3.12+ and ffmpeg

# Clone and install
cd audiobook-creator
uv venv
.venv\Scripts\activate  # Windows
uv pip install -e .

# Install FFmpeg (if not already in PATH)
# Windows: winget install Gyan.FFmpeg
# Linux: sudo apt-get install ffmpeg
# Mac: brew install ffmpeg
```

### Model Setup

#### 1. Supertonic (Default)
```bash
# Download models (one-time setup)
git clone https://huggingface.co/Supertone/supertonic %USERPROFILE%\.cache\huggingface\supertonic_models
```

#### 2. Kokoro / Chatterbox
These models are automatically downloaded via the `transformers` or `kokoro` libraries when first used.

### Basic Usage

```bash
# Generate audiobook (Default: Supertonic, voice M1)
audiobook-creator --input book.md --output ./audiobook

# Use Kokoro model
audiobook-creator --input book.md --model kokoro --voice-style af

# Use Chatterbox model
audiobook-creator --input book.md --model chatterbox

# Use caching to skip existing chapters
audiobook-creator --input book.md --use-cache

# Create single concatenated file
audiobook-creator --input book.md --concat
```

## CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `--input`, `-i` | Input Markdown file (required) | - |
| `--output`, `-o` | Output directory | `./output` |
| `--model` | TTS Model (`supertonic`, `kokoro`, `chatterbox`) | `supertonic` |
| `--voice-style` | Voice style (Model dependent) | `default` |
| `--format`, `-f` | Audio format (`wav`, `mp3`) | `mp3` |
| `--ffmpeg-path` | Manual path to ffmpeg executable | (Auto-detected) |
| `--use-cache` | Skip generation for existing files | `False` |
| `--concat` | Generate single full audiobook file | `False` |
| `--verbose`, `-v` | Enable verbose logging | `False` |
| `--no-dynamic-pauses` | Disable semantic similarity pauses | `False` |

## Voice Styles

- **Supertonic**: `M1`-`M3` (Male), `F1`-`F3` (Female).
- **Kokoro**: `af` (Default), `am`, `bf`, `bm`, etc.
- **Chatterbox**: Uses model defaults.

## Chapter Detection

Automatically detects first-level Markdown headings (`#`):

```markdown
# Chapter 1: Introduction
# Chapter 2: Main Content  
# Preface
```

The tool also fixes duplicate title narration and adds a 1.5s pause after headings.

## How It Works

1. **Parse**: Extract chapters from Markdown (fixes generic H1 detection).
2. **Chunk**: Split text into optimal TTS chunks.
3. **Synthesize**: Generate audio using the selected model.
4. **Pause**: Calculate semantic similarity for natural pauses (1.5s for headings).
5. **Convert**: Output as MP3 using **in-memory processing** for maximum speed.

## Global Audience & Character Support

Audiobook Creator is designed for a global audience:
- **Unicode Support**: Full support for non-ASCII characters and diacritics.
- **Recursive Splitting**: Handles extremely long sentences or words (like in German or technical texts) by recursively splitting them to fit model constraints.
- **Multi-Model**: Choose models that best fit your language's phonetics (e.g., Kokoro has excellent multi-language potential).

## Troubleshooting

### FFmpeg Not Found
If the tool can't find FFmpeg, it will provide a direct `winget` command to install it. You can also specify the path manually using `--ffmpeg-path`.

### Unicode Encoding Errors
If you see `UnicodeEncodeError` in Windows console:
```powershell
$env:PYTHONUTF8 = "1"
```

## Project Structure

```
audiobook-creator/
├── src/audiobook_creator/
│   ├── parser.py           # Markdown parsing (H1 fix)
│   ├── chunker.py          # Text chunking
│   ├── tts_engine.py        # Modular TTS interface (Factory)
│   ├── supertonic_wrapper.py # Supertonic integration (Strict 300-char limit)
│   ├── dynamic_pause.py    # Semantic pause calculation
│   ├── audiobook.py        # Audio generation (In-memory optimization & Caching)
│   └── cli.py              # Command-line interface (New args)
├── books/                   # Sample books
├── pyproject.toml
└── README.md
```

## License

MIT License

## Acknowledgements

- **Supertone** for Supertonic TTS
- **hexgrad** for Kokoro TTS
- **Resemble AI** for Chatterbox-Turbo
- **Hugging Face** for model hosting
- **ONNX Runtime** for fast CPU inference

---

**Built by Amit Kumar with ❤️**