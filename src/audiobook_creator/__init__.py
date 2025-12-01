"""
Audiobook Creator Package

Convert Markdown books to audiobooks using Supertone Supertonic TTS.
"""

__version__ = "1.0.0"

from .parser import MarkdownParser, Chapter
from .chunker import TextChunker
from .tts_engine import TTSEngine
from .audiobook import AudiobookGenerator

__all__ = [
    "MarkdownParser",
    "Chapter",
    "TextChunker",
    "TTSEngine",
    "AudiobookGenerator"
]
