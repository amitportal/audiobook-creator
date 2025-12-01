"""
Audiobook Generator Module

Orchestrates chapter-wise audio generation and concatenation.
"""

import logging
from pathlib import Path
from typing import List, Optional
import numpy as np
import soundfile as sf
from tqdm import tqdm

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    AudioSegment = None

from .parser import Chapter, MarkdownParser
from .chunker import TextChunker
from .tts_engine import TTSEngine
from .dynamic_pause import DynamicPauseCalculator


logger = logging.getLogger(__name__)


class AudiobookGenerator:
    """Generate audiobook files from parsed Markdown chapters."""
    
    def __init__(
        self,
        tts_engine: TTSEngine,
        chunker: TextChunker,
        output_dir: Path = Path("./output"),
        audio_format: str = "mp3",
        use_dynamic_pauses: bool = True
    ):
        """
        Initialize audiobook generator.
        
        Args:
            tts_engine: Initialized TTS engine
            chunker: Text chunker instance
            output_dir: Output directory for audio files
            audio_format: Audio format (wav or mp3)
            use_dynamic_pauses: Use semantic similarity for dynamic pauses
        """
        self.tts_engine = tts_engine
        self.chunker = chunker
        self.output_dir = Path(output_dir)
        self.audio_format = audio_format
        self.use_dynamic_pauses = use_dynamic_pauses
        
        # Initialize dynamic pause calculator if enabled
        if use_dynamic_pauses:
            self.pause_calculator = DynamicPauseCalculator()
        else:
            self.pause_calculator = None
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Audiobook generator initialized")
        logger.info(f"  Output directory: {self.output_dir}")
        logger.info(f"  Audio format: {self.audio_format}")
        logger.info(f"  Dynamic pauses: {'enabled' if use_dynamic_pauses else 'disabled'}")
    
    def generate_audiobook(
        self,
        markdown_file: Path,
        concatenate: bool = False
    ) -> List[Path]:
        """
        Generate audiobook from a Markdown file.
        
        Args:
            markdown_file: Path to Markdown book file
            concatenate: Whether to create a full audiobook file
            
        Returns:
            List of generated audio file paths
        """
        logger.info(f"Starting audiobook generation from {markdown_file}")
        
        # Parse the book
        parser = MarkdownParser(markdown_file)
        chapters = parser.parse()
        
        logger.info(f"Found {len(chapters)} chapters")
        
        # Generate chapter audios
        chapter_files = []
        
        for i, chapter in enumerate(tqdm(chapters, desc="Generating chapters")):
            logger.info(f"Processing chapter {i+1}/{len(chapters)}: {chapter.title}")
            
            chapter_file = self.generate_chapter_audio(chapter)
            chapter_files.append(chapter_file)
        
        # Concatenate if requested
        if concatenate and len(chapter_files) > 1:
            logger.info("Creating full audiobook file")
            full_audiobook = self.concatenate_chapters(chapter_files)
            chapter_files.append(full_audiobook)
        
        logger.info(f"Audiobook generation complete. {len(chapter_files)} files created.")
        return chapter_files
    
    def generate_chapter_audio(self, chapter: Chapter) -> Path:
        """
        Generate audio for a single chapter with dynamic pauses.
        
        Args:
            chapter: Chapter object
            
        Returns:
            Path to generated audio file
        """
        # Chunk the chapter
        chunks = self.chunker.chunk_chapter(chapter)
        
        logger.info(f"Chapter '{chapter.title}' split into {len(chunks)} chunks")
        
        # Generate audio for each chunk with dynamic pauses
        chunk_audios = []
        
        for i, chunk in enumerate(tqdm(chunks, desc=f"  Chunks", leave=False)):
            audio = self.tts_engine.synthesize(chunk.text)
            chunk_audios.append(audio)
            
            # Add pause between chunks (except after last chunk)
            if i < len(chunks) - 1:
                # Calculate pause duration
                if self.use_dynamic_pauses and self.pause_calculator:
                    try:
                        pause_duration = self.pause_calculator.calculate_pause(
                            chunk.text,
                            chunks[i + 1].text,
                            default_pause=0.3
                        )
                    except Exception as e:
                        logger.debug(f"Dynamic pause calculation failed, using default: {e}")
                        pause_duration = 0.3
                else:
                    pause_duration = 0.3  # Default pause
                
                # Create pause
                pause_samples = int(self.tts_engine.sample_rate * pause_duration)
                pause = np.zeros(pause_samples, dtype=np.float32)
                chunk_audios.append(pause)
        
        # Concatenate all audios
        chapter_audio = np.concatenate(chunk_audios)
        
        # Generate filename
        filename = self._get_chapter_filename(chapter)
        output_path = self.output_dir / filename
        
        # Save audio
        self._save_audio(chapter_audio, output_path)
        
        logger.info(f"Chapter audio saved: {output_path}")
        return output_path
    
    def concatenate_chapters(self, chapter_files: List[Path]) -> Path:
        """
        Concatenate chapter audio files into a full audiobook.
        
        Args:
            chapter_files: List of chapter audio file paths
            
        Returns:
            Path to full audiobook file
        """
        if not PYDUB_AVAILABLE and self.audio_format == "mp3":
            logger.warning("pydub not available, falling back to WAV format for concatenation")
            self.audio_format = "wav"
        
        logger.info(f"Concatenating {len(chapter_files)} chapter files")
        
        if self.audio_format == "wav" or not PYDUB_AVAILABLE:
            # Use numpy/soundfile for WAV
            combined_audio = []
            silence = np.zeros(int(self.tts_engine.sample_rate * 2.0), dtype=np.float32)  # 2s silence
            
            for chapter_file in tqdm(chapter_files, desc="Concatenating"):
                audio, sr = sf.read(str(chapter_file))
                if combined_audio:
                    combined_audio.append(silence)
                combined_audio.append(audio)
            
            final_audio = np.concatenate(combined_audio)
            filename = f"Full_Audiobook.wav"
            output_path = self.output_dir / filename
            sf.write(output_path, final_audio, self.tts_engine.sample_rate)
        else:
            # Use pydub for MP3
            combined = None
            
            for chapter_file in tqdm(chapter_files, desc="Concatenating"):
                audio = AudioSegment.from_file(str(chapter_file))
                
                if combined is None:
                    combined = audio
                else:
                    # Add silence between chapters
                    silence = AudioSegment.silent(duration=2000)  # 2 seconds
                    combined = combined + silence + audio
            
            # Save full audiobook
            filename = f"Full_Audiobook.{self.audio_format}"
            output_path = self.output_dir / filename
            
            combined.export(str(output_path), format=self.audio_format)
        
        logger.info(f"Full audiobook saved: {output_path}")
        return output_path
    
    def _get_chapter_filename(self, chapter: Chapter) -> str:
        """Generate filename for a chapter."""
        if chapter.number:
            filename = f"Chapter_{chapter.number:02d}_{chapter.filename_safe_title}"
        else:
            filename = f"Chapter_{chapter.filename_safe_title}"
        
        return f"{filename}.{self.audio_format}"
    
    def _save_audio(self, audio: np.ndarray, output_path: Path) -> None:
        """Save audio array to file."""
        if self.audio_format == "wav":
            sf.write(output_path, audio, self.tts_engine.sample_rate)
        else:  # mp3
            if not PYDUB_AVAILABLE:
                logger.warning("pydub not available, saving as WAV instead of MP3")
                output_path = output_path.with_suffix('.wav')
                sf.write(output_path, audio, self.tts_engine.sample_rate)
            else:
                # Save as WAV first, then convert
                temp_wav = output_path.with_suffix('.wav')
                sf.write(temp_wav, audio, self.tts_engine.sample_rate)
                
                # Convert to MP3
                audio_segment = AudioSegment.from_wav(str(temp_wav))
                audio_segment.export(str(output_path), format="mp3")
                
                # Remove temp file
                temp_wav.unlink()
