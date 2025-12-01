"""
Text Chunking Module

Splits chapter text into TTS-friendly chunks with smart boundary detection.
"""

import re
from dataclasses import dataclass
from typing import List
from .parser import Chapter


@dataclass
class TextChunk:
    """Represents a chunk of text ready for TTS processing."""
    text: str
    chapter_number: int | None
    chapter_title: str
    sequence: int
    is_heading: bool = False


class TextChunker:
    """Chunks text for optimal TTS processing."""
    
    def __init__(
        self,
        max_chunk_size: int = 1000,
        long_paragraph_threshold: int = 1500,
        include_headings: bool = True
    ):
        """
        Initialize the chunker.
        
        Args:
            max_chunk_size: Maximum characters per chunk
            long_paragraph_threshold: Threshold to trigger sentence-based splitting
            include_headings: Whether to narrate chapter headings
        """
        self.max_chunk_size = max_chunk_size
        self.long_paragraph_threshold = long_paragraph_threshold
        self.include_headings = include_headings
    
    def chunk_chapter(self, chapter: Chapter) -> List[TextChunk]:
        """
        Chunk a chapter into TTS-ready segments.
        
        Args:
            chapter: Chapter object to chunk
            
        Returns:
            List of TextChunk objects
        """
        chunks = []
        sequence = 0
        
        # Add heading as first chunk if enabled
        if self.include_headings:
            heading_text = self._format_heading(chapter)
            chunks.append(TextChunk(
                text=heading_text,
                chapter_number=chapter.number,
                chapter_title=chapter.title,
                sequence=sequence,
                is_heading=True
            ))
            sequence += 1
        
        # Process paragraphs
        for paragraph in chapter.paragraphs:
            # Skip if paragraph is just the heading
            if self._is_heading_paragraph(paragraph, chapter):
                continue
            
            # Chunk the paragraph
            para_chunks = self._chunk_paragraph(paragraph)
            
            for chunk_text in para_chunks:
                chunks.append(TextChunk(
                    text=chunk_text,
                    chapter_number=chapter.number,
                    chapter_title=chapter.title,
                    sequence=sequence,
                    is_heading=False
                ))
                sequence += 1
        
        return chunks
    
    def _format_heading(self, chapter: Chapter) -> str:
        """Format chapter heading for narration."""
        if chapter.number:
            return f"Chapter {chapter.number}. {chapter.title}."
        else:
            return f"{chapter.title}."
    
    def _is_heading_paragraph(self, paragraph: str, chapter: Chapter) -> bool:
        """Check if paragraph is the chapter heading itself."""
        # More precise matching: only skip if it's EXACTLY the heading
        # Not if it just contains the heading word somewhere
        para_stripped = paragraph.strip()
        
        # Check if it's just the heading (possibly with the # prefix)
        if para_stripped == chapter.title:
            return True
        
        # Check if it's the markdown heading line
        if para_stripped == f"# {chapter.title}":
            return True
        
        # Check if it's a very short paragraph that's just the title (within 10 chars)
        if len(para_stripped) < len(chapter.title) + 10:
            para_clean = re.sub(r'\W+', '', para_stripped.lower())
            title_clean = re.sub(r'\W+', '', chapter.title.lower())
            if para_clean == title_clean:
                return True
        
        return False
    
    def _chunk_paragraph(self, paragraph: str) -> List[str]:
        """
        Chunk a single paragraph.
        
        Uses paragraph-based chunking by default, falls back to sentence-based
        for long paragraphs.
        """
        # If paragraph is short enough, return as-is
        if len(paragraph) <= self.max_chunk_size:
            return [paragraph]
        
        # If paragraph is very long, split by sentences
        if len(paragraph) > self.long_paragraph_threshold:
            return self._chunk_by_sentences(paragraph)
        
        # Otherwise, try to split at natural boundaries (but may exceed max_chunk_size)
        # This handles moderately long paragraphs
        return [paragraph]
    
    def _chunk_by_sentences(self, text: str) -> List[str]:
        """Split text into chunks by sentence boundaries."""
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            sentence_length = len(sentence)
            
            # If adding this sentence would exceed limit, save current chunk
            if current_length + sentence_length > self.max_chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length + 1  # +1 for space
        
        # Add remaining chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def chunk_chapters(self, chapters: List[Chapter]) -> dict[str, List[TextChunk]]:
        """
        Chunk multiple chapters.
        
        Args:
            chapters: List of Chapter objects
            
        Returns:
            Dictionary mapping chapter identifiers to chunk lists
        """
        result = {}
        
        for chapter in chapters:
            # Create unique key for chapter
            if chapter.number:
                key = f"chapter_{chapter.number:02d}"
            else:
                key = f"chapter_{chapter.filename_safe_title.lower()}"
            
            result[key] = self.chunk_chapter(chapter)
        
        return result
