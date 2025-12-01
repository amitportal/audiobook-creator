"""
Command Line Interface for Audiobook Creator

Simplified for Supertonic TTS only.
"""

import argparse
import logging
import sys
from pathlib import Path

from . import __version__
from .audiobook import AudiobookGenerator
from .tts_engine import TTSEngine
from .chunker import TextChunker


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('audiobook_creation.log')
        ]
    )


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Convert Markdown books to audiobooks using Supertone Supertonic TTS',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate audiobook with default settings
  audiobook-creator --input book.md

  # With custom output directory
  audiobook-creator --input book.md --output ./my_audiobook

  # Create full audiobook file  
  audiobook-creator --input book.md --concat

  # Verbose logging
  audiobook-creator --input book.md --verbose

TTS Model:
  Supertone Supertonic - ONNX-based, ultra-fast, high-quality
  Sample rate: 44100 Hz | Speed: 4.4x real-time
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Input Markdown file path'
    )
    
    # Optional arguments
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='./output',
        help='Output directory (default: ./output)'
    )
    
    parser.add_argument(
        '--format', '-f',
        type=str,
        choices=['wav', 'mp3'],
        default='mp3',
        help='Output audio format (default: mp3, requires ffmpeg)'
    )
    
    parser.add_argument(
        '--concat',
        action='store_true',
        help='Generate full audiobook file (concatenates all chapters)'
    )
    
    parser.add_argument(
        '--voice-style',
        type=str,
        choices=['M1', 'M2', 'M3', 'F1', 'F2', 'F3'],
        default='M1',
        help='Voice style: M1-M3 (male), F1-F3 (female) (default: M1)'
    )
    
    parser.add_argument(
        '--no-dynamic-pauses',
        action='store_true',
        help='Disable semantic similarity-based dynamic pauses'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version=f'%(prog)s {__version__}'
    )
    
    # Chunking parameters
    parser.add_argument(
        '--max-chunk-size',
        type=int,
        default=1000,
        help='Maximum characters per chunk (default: 1000)'
    )
    
    parser.add_argument(
        '--no-headings',
        action='store_true',
        help='Do not narrate chapter headings'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Audiobook Creator v{__version__}")
    logger.info(f"Input file: {args.input}")
    logger.info(f"Output directory: {args.output}")
    logger.info(f"TTS Model: Supertone Supertonic (ONNX)")
    logger.info(f"Format: {args.format}")
    
    # Validate input file
    input_file = Path(args.input)
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        sys.exit(1)
    
    if not input_file.suffix.lower() == '.md':
        logger.warning(f"Input file does not have .md extension: {input_file}")
    
    try:
        # Initialize components
        logger.info("Initializing Supertonic TTS engine...")
        tts_engine = TTSEngine(device="cpu", voice_style=args.voice_style)
        
        logger.info(f"Loading TTS model with voice style '{args.voice_style}'...")
        tts_engine.load_model()
        
        logger.info("Initializing text chunker...")
        chunker = TextChunker(
            max_chunk_size=args.max_chunk_size,
            include_headings=not args.no_headings
        )
        
        logger.info("Initializing audiobook generator...")
        generator = AudiobookGenerator(
            tts_engine=tts_engine,
            chunker=chunker,
            output_dir=Path(args.output),
            audio_format=args.format,
            use_dynamic_pauses=not args.no_dynamic_pauses
        )
        
        # Generate audiobook
        logger.info("Starting audiobook generation...")
        output_files = generator.generate_audiobook(
            markdown_file=input_file,
            concatenate=args.concat
        )
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("Audiobook generation completed successfully!")
        logger.info(f"Generated {len(output_files)} file(s):")
        for file in output_files:
            file_size = file.stat().st_size / (1024 * 1024)  # MB
            logger.info(f"  - {file.name} ({file_size:.2f} MB)")
        logger.info("="*60)
        
    except KeyboardInterrupt:
        logger.warning("\nGeneration interrupted by user")
        sys.exit(130)
    
    except Exception as e:
        logger.error(f"Error during audiobook generation: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
