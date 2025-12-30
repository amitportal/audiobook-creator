# Audiobook Creator Usage Guide

This guide provides a deep-dive into how to get the most out of **Audiobook Creator**. Whether you're a casual listener or a power user, these tips will help you create professional-grade audiobooks.

---

## 📖 1. Preparing Your Content

Audiobook Creator works best with structured Markdown. 

### Markdown Best Practices
*   **Use Headers**: Use `#` (H1) for Chapter titles. The parser uses these to split the book into logical audio files.
*   **Clean Text**: Avoid complex tables or raw code blocks as they don't always translate well to speech.
*   **Semantic Indicators**: The "Dynamic Pauses" feature looks for sentences semantic differences, paragraph breaks, and punctuation to inject natural timing.

---

## 🛠️ 2. Using the GUI

The graphical interface is designed to be self-explanatory but here are some powerful features:

1.  **Input File**: Select any `.md` file.
2.  **Output Directory**: Where your MP3/WAV files will be saved.
3.  **TTS Model**:
    *   **Supertonic**: The all-rounder. Fast, reliable, and sounds like a professional narrator.
    *   **Soprano**: Ultra-lightweight. Perfect for older laptops or when you want quick results.
    *   **MiraTTS**: High-fidelity (48kHz). Best for when you want the absolute highest "human-like" quality.
    *   **Kokoro**: Excellent for expressive, melodic speech.
    *   **Chatterbox**: Best for dialogue-heavy books.
4.  **Voice Style**: Use `M1-M3` (Male) or `F1-F3` (Female) for Supertonic. Other models have their own defaults (e.g., `af` for Kokoro).
5.  **Concatenate Output**: If checked, you'll get one massive MP3 file. If unchecked, you get one file per chapter.

---

## 💻 3. Command Line Interface (CLI)

For batch processing or automation:

```bash
# General Syntax
uv run audiobook -i <input.md> -o <output_dir> [options]

# Example: Create a single audiobook with MiraTTS on CUDA
uv run audiobook -i my_book.md -o ./finished -m miratts --concat --device cuda
```

### Advanced CLI Flags
*   `--use-cache`: Skip generating chapters that haven't changed.
*   `--no-dynamic-pauses`: If you prefer a constant speed without semantic pauses.
*   `--verbose`: See exactly what the models are "thinking" during generation.

---

## 🚀 4. Hardware Tuning

### NVIDIA Users (CUDA)
*   **Performance**: Supertonic and Soprano will be significantly faster.
*   **Recommendation**: Always use `--device cuda` if you have 8GB+ VRAM.

### Intel Users (OpenVINO / NPU)
*   **Efficiency**: If you have a Core Ultra chip, Supertonic and Chatterbox will run on the NPU, saving your battery while keeping the CPU free for other tasks.
*   **Automatic**: Our `hardware.py` utility handles this automatically.

---

## 💡 5. Pro Tips for Better Books

1.  **Test a Single Chapter first**: Don't run the whole book immediately. Verify the voice and speed on Chapter 1.
2.  **Use Caching**: If you make a small edit to Chapter 5, enabling `--use-cache` will ensure Chapters 1-4 aren't re-generated.
3.  **Voice Matching**: 
    *   Use **Supertonic** for Non-Fiction (Clear, steady).
    *   Use **Chatterbox** or **MiraTTS** for Fiction (More emotion and dynamic range).
4.  **Audio Fixes**: If a sentence sounds "clunky", try adding a comma or splitting it into two. Transformer based models often react better to standard punctuation.

---

## ❓ 6. Troubleshooting

| Issue | Solution |
|-------|----------|
| **Robotic Sound** | Try lowering the complexity of the sentence or switching to **MiraTTS**. |
| **Out of Memory** | Ensure you aren't running other heavy browser tabs. Use **Soprano-80M** for the lowest memory footprint. |
| **FFmpeg Error** | The setup script usually fixes this. If not, install via `winget install Gyan.FFmpeg`. |

---

*Built with ❤️ for the Listeners, Learners, and Readers. Please upload good audiobooks conversions to Librivox or other open community projects. If you find this useful, please consider supporting me on [GitHub Sponsors](https://github.com/sponsors/amitportal).*

*Contributions are welcome! Please open an issue or PR for new model integrations or hardware optimizations.*
