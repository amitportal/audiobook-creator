# Audiobook Creator

**Version 1.2.0** - Convert Markdown books into high-quality MP3 audiobooks with state-of-the-art TTS models and hardware acceleration.

[![Software License](https://img.shields.io/badge/license-AGPL_V3-blue.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Hardware Support](https://img.shields.io/badge/hardware-NVIDIA_CUDA_|_Intel_NPU_|_CPU-green.svg)](#hardware-acceleration)

## Features

- **Multi-Model Support**:
  - **Supertone Supertonic**: Ultra-fast ONNX-based high-fidelity models.
  - **Soprano-80M**: Lightweight LLM-based TTS with Vocos decoder.
  - **MiraTTS**: High-quality 48kHz audio tokens with NCodec.
  - **Kokoro**: High-quality open-weight model for expressive speech.
  - **Chatterbox**: Paralinguistic support for natural sounding books.
- **Hardware Acceleration**: Automatic detection of **NVIDIA GPUs (CUDA)**, **Intel NPUs/GPUs (OpenVINO)**, and **DirectML**.
- **Smart Parsing**: Automatically handles Markdown structures, headers, and semantic pauses.
- **Dynamic Pauses**: Uses semantic similarity to inject natural breathing room between sections.
- **One-Click Setup**: Professional PowerShell script handles all dependencies, models, and FFmpeg automatically.
- **Modern GUI**: dark-mode interface with real-time processing logs.

## Quick Start (Windows)

1. **Clone/Download** this repository.
2. **Right-click** `run_audiobook_creator.ps1` and select **"Run with PowerShell"**.
3. **Wait** for the automatic setup (Installs UV, FFmpeg, and Models).
4. **The GUI launches automatically!**

## Hardware Acceleration

Audiobook Creator automatically optimizes itself for your hardware:

| Hardware Detect | Method Used | Benefit |
|-----------------|-------------|---------|
| **NVIDIA GPU** | CUDA 12.x | Maximum performance for all models. |
| **Intel NPU** | OpenVINO | Efficient background processing on modern laptops. |
| **Intel GPU** | OpenVINO | High-performance inference. |
| **Old GPU/AMD** | DirectML | Basic hardware acceleration. |
| **Generic CPU** | AVX/AVX2 | Optimized fallback for all systems. |

## Usage

### GUI
Simply select your input file, choose a model, and click "Start".

### Command Line
```bash
# Basic usage
uv run audiobook -i book.md -m soprano

# Specify voice and hardware
uv run audiobook -i book.md -m supertonic -v M2 --device cuda
```

## Supported Models

| Model | Sample Rate | Best For | Hardware |
|-------|-------------|----------|----------|
| **Supertonic** | 44.1 kHz | Speed & Stability | ONNX (All) |
| **Soprano** | 32.0 kHz | Efficiency (80M params) | Torch / CUDA |
| **MiraTTS** | 48.0 kHz | Ultra Fidelity | Torch / CUDA |
| **Kokoro** | 24.0 kHz | Quality | Torch / CPU |
| **Chatterbox** | 24.0 kHz | Expression | ONNX (All) |

---

## Technical Setup

### Prerequisites
- Python 3.12+
- FFmpeg (Auto-installed by script)

### Manual Installation
```bash
pip install uv
uv sync
uv pip install -e .
audiobook-gui
```

## Troubleshooting

- **Security Policy**: If PowerShell scripts are blocked, run:
  `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`
- **Missing Models**: Ensure you have ~2GB free space for one-time model downloads.
- **FFmpeg Output**: If MP3 conversion fails, ensure FFmpeg is in your system PATH.

## License

This project is licensed under the AGPL-3.0 License - see the [LICENSE](LICENSE) file for details.

## Credits

Built with ❤️ by [Amit Kumar](https://github.com/amitportal). 

Contributions are welcome! Please open an issue or PR for new model integrations or hardware optimizations.