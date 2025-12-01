"""
Markdown Parser Module

Parses Markdown files, detects chapter boundaries, and extracts clean text content.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import markdown
from markdown.extensions import Extension
from markdown.treeprocessors import Treeprocessor


@dataclass
class Chapter:
    """Represents a book chapter."""
    number: Optional[int]
    title: str
    content: str
    paragraphs: List[str]
    
    @property
    def filename_safe_title(self) -> str:
        """Get a filesystem-safe version of the title."""
        # Remove special characters and replace spaces with underscores
        safe = re.sub(r'[^\w\s-]', '', self.title)
        safe = re.sub(r'[-\s]+', '_', safe)
        return safe.strip('_')


class PlainTextTreeprocessor(Treeprocessor):
    """Extract plain text from Markdown ElementTree."""
    
    def run(self, root):
        self.text = self._get_text(root)
        return root
    
    def _get_text(self, elem):
        """Recursively extract text from element."""
        text = elem.text or ''
        for child in elem:
            text += self._get_text(child)
            text += child.tail or ''
        return text


class PlainTextExtension(Extension):
    """Markdown extension to extract plain text."""
    
    def extendMarkdown(self, md):
        md.registerExtension(self)
        self.processor = PlainTextTreeprocessor(md)
        md.treeprocessors.register(self.processor, 'plaintext', 0)


class MarkdownParser:
    """Parser for Markdown book files."""
    
    # Patterns for chapter detection
    CHAPTER_PATTERNS = [
        r'^#\s+Chapter\s+(\d+)\s*(.*)$',  # # Chapter 1 Title
        r'^#\s+(Preface|Acknowledgements|Epilogue|Foreword|Introduction|Conclusions?)$',
        r'^#\s+(Endorsements?|Contents?)$',
    ]
    
    def __init__(self, filepath: Path):
        """Initialize parser with a Markdown file path."""
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
    
    def parse(self) -> List[Chapter]:
        """Parse the Markdown file and extract chapters."""
        with open(self.filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        chapters = []
        lines = content.split('\n')
        current_chapter_lines = []
        current_chapter_info = None
        
        for line in lines:
            chapter_info = self._detect_chapter_heading(line)
            
            if chapter_info:
                # Save previous chapter if exists
                if current_chapter_info and current_chapter_lines:
                    chapter = self._build_chapter(
                        current_chapter_info,
                        '\n'.join(current_chapter_lines)
                    )
                    chapters.append(chapter)
                
                # Start new chapter
                current_chapter_info = chapter_info
                current_chapter_lines = [line]  # Include heading
            elif current_chapter_info:
                current_chapter_lines.append(line)
        
        # Don't forget the last chapter
        if current_chapter_info and current_chapter_lines:
            chapter = self._build_chapter(
                current_chapter_info,
                '\n'.join(current_chapter_lines)
            )
            chapters.append(chapter)
        
        return chapters
    
    def _detect_chapter_heading(self, line: str) -> Optional[dict]:
        """Detect if a line is a chapter heading."""
        for pattern in self.CHAPTER_PATTERNS:
            match = re.match(pattern, line.strip(), re.IGNORECASE)
            if match:
                groups = match.groups()
                
                # Pattern 1: # Chapter N Title
                if len(groups) == 2 and groups[0].isdigit():
                    return {
                        'number': int(groups[0]),
                        'title': groups[1].strip() or f"Chapter {groups[0]}"
                    }
                
                # Pattern 2 & 3: # Special Chapter Name
                elif len(groups) == 1:
                    return {
                        'number': None,
                        'title': groups[0].strip()
                    }
        
        return None
    
    def _build_chapter(self, chapter_info: dict, raw_content: str) -> Chapter:
        """Build a Chapter object from raw content."""
        # Clean the content
        cleaned_content = self._clean_markdown(raw_content)
        
        # Split into paragraphs
        paragraphs = self._extract_paragraphs(cleaned_content)
        
        return Chapter(
            number=chapter_info['number'],
            title=chapter_info['title'],
            content=cleaned_content,
            paragraphs=paragraphs
        )
    
    def _clean_markdown(self, text: str) -> str:
        """Remove Markdown formatting and clean text."""
        # Convert Markdown to plain text
        md = markdown.Markdown(extensions=[PlainTextExtension()])
        md.convert(text)
        plain_text = md.treeprocessors.get_index_for_name('plaintext')
        if plain_text >= 0:
            processor = md.treeprocessors[plain_text]
            text = processor.text
        
        # Additional cleaning
        # Remove multiple spaces
        text = re.sub(r' +', ' ', text)
        
        # Remove excessive newlines (more than 2)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Normalize quotation marks
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def _extract_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs."""
        # Split on double newlines
        paragraphs = re.split(r'\n\s*\n', text)
        
        # Clean and filter empty paragraphs
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        return paragraphs
