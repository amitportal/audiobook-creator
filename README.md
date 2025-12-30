# Audiobook Creator

**Version 1.1.0** - Convert Markdown books into high-quality MP3 audiobooks using multiple TTS models (Supertonic, Kokoro, Chatterbox).

## Features
- **Modular TTS Architecture**: Support for multiple TTS backends:
- **Supertone Supertonic** (Default): Ultra-fast ONNX-based models.
- **Kokoro**: Lightweight, high-quality open-weight model.
- **ResembleAI Chatterbox**: Expressive model with paralinguistic tag support.
- **Smart Chapter Detection**: Automatically detects chapters from Markdown headings (including generic H1).
- **MP3 Output**: Professional MP3 audiobooks with automatic conversion.
- **Automatic FFmpeg Detection**: Robust search for FFmpeg on Windows (WinGet, Program Files, etc.).
- **Performance Optimized**: In-memory audio processing (no intermediate WAV files).
- **Smart Caching**: Skip re-generating chapters that already have an output file with `--use-cache`.
- **Dynamic Pauses**: Semantic similarity-based pauses between text chunks for natural flow.
- **Clear Progress**: Real-time console output "yelling" the current chapter being processed.
- **Configurable Voice Styles**: Choose different voice styles depending on the model.
- **Modern GUI**: Graphical interface with dark mode, file pickers, and real-time logs.
- **One-Click Deployment**: Double-click PowerShell script handles everything.

## Quick Start

### Easiest Method: One-Click Script (Windows)

**Requirements:** Just Python 3.12+ installed on your PC

**Steps:**
1. Download or clone this repository
2. Right-click `run_audiobook_creator.ps1`
3. Select "Run with PowerShell"
4. Wait for automatic setup (first run only)
5. GUI launches automatically!

The script automatically:
- Checks Python version
- Installs `uv` package manager
- Installs all dependencies
- Installs FFmpeg (if not present)
- Downloads Supertonic models (~500MB, one-time)
- Launches the GUI

**Note:** If you encounter a security warning:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

### Alternative: Manual GUI Setup

For users who prefer manual control or are using Linux/Mac:

```bash
# Prerequisites: Python 3.12+ and git

# Step 1: Install uv
# Windows:
winget install astral-sh.uv
# Linux/Mac:
curl -LsSf https://astral.sh/uv/install.sh | sh

# Step 2: Clone and setup
git clone https://github.com/amitportal/audiobook-creator.git
cd audiobook-creator
uv sync
uv pip install -e .
3. **Model Settings**: Select TTS model and voice (`M1`-`M3` for male, `F1`-`F3` for female)
4. **Options**: Enable cache, concatenated output, or dynamic pauses
5. **Start**: Click "Start Generation" and monitor real-time logs!

---

### Command Line Interface

For automation or advanced users:

```bash
# Basic usage
uv run audiobook -i book.md -o ./audiobook

# With specific model
uv run audiobook -i book.md -m kokoro -v af

# Use caching
uv run audiobook -i book.md --use-cache

# Create single file
uv run audiobook -i book.md --concat
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

- **Supertonic**: `M1`-`M3` (Male), `F1`-`F3` (Female)
- **Kokoro**: `af` (Default), `am`, `bf`, `bm`, etc.
- **Chatterbox**: Uses model defaults

## How It Works

1. **Parse**: Extract chapters from Markdown (supports generic H1)
2. **Chunk**: Split text into optimal TTS chunks (model-specific limits)
3. **Synthesize**: Generate audio using selected model
4. **Pause**: Calculate semantic similarity for natural pauses (1.5s for headings)
5. **Convert**: Output as MP3 using in-memory processing

## Troubleshooting

### Script Security Warning
If Windows blocks the script:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### FFmpeg Not Found
The script auto-installs via winget. Manual install:
```powershell
winget install Gyan.FFmpeg
```

### GUI Command Not Found
Reinstall the package:
```powershell
uv pip install -e .
uv run audiobook-gui
```

### Unicode Encoding Errors
If you see `UnicodeEncodeError`:
```powershell
$env:PYTHONUTF8 = "1"
```

## Deployment to Other PCs

**Requirements:** Python 3.12+ installed

**Method 1 (Easiest):** Share the entire `audiobook-creator` folder and double-click `run_audiobook_creator.ps1`

**Method 2 (Manual):** Clone repo and run:
```powershell
uv sync
uv pip install -e .
uv run audiobook-gui
```

## Project Structure

```
audiobook-creator/
├── run_audiobook_creator.ps1  # One-click setup & launch script
├── src/audiobook_creator/
│   ├── parser.py              # Markdown parsing
│   ├── chunker.py             # Text chunking
│   ├── tts_engine.py          # Modular TTS interface
│   ├── supertonic_wrapper.py  # Supertonic ONNX integration
│   ├── dynamic_pause.py       # Semantic pause calculation
│   ├── audiobook.py           # Audio generation
│   ├── gui.py                 # Modern GUI application
│   └── cli.py                 # Command-line interface
├── books/                      # Sample books
├── pyproject.toml
└── README.md
```

## License

GNU Affero General Public License (AGPL V3)

## Acknowledgements

- **Supertone** for Supertonic TTS
- **hexgrad** for Kokoro TTS
- **Resemble AI** for Chatterbox-Turbo
- **Hugging Face** for model hosting
- **ONNX Runtime** for fast CPU inference
- **CustomTkinter** for the modern GUI

---

**Built by Amit Kumar with ❤️**