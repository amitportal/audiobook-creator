"""
TTS Engine Module - Supertonic ONLY

Simplified implementation using only Supertone Supertonic ONNX model.
"""

import logging
from pathlib import Path
from typing import Optional
import numpy as np
import soundfile as sf

from .supertonic_wrapper import SupertonicTTS


logger = logging.getLogger(__name__)


class TTSEngine:
    """Text-to-Speech engine using Supertone Supertonic."""
    
    def __init__(self, device: str = "cpu", voice_style: str = "M1"):
        """
        Initialize TTS engine with Supertonic.
        
        Args:
            device: Device preference (Supertonic uses CPU-optimized ONNX)
            voice_style: Voice style identifier (M1-M3, F1-F3)
        """
        self.model = None
        self.device = device
        self.voice_style = voice_style
        self.sample_rate = 44100  # Supertonic sample rate
        
        logger.info("Initializing Supertonic TTS engine")
        logger.info(f"Voice style: {voice_style}")
        logger.info("Description: Supertone Supertonic - ONNX based, ultra-fast")
        logger.info(f"Sample rate: {self.sample_rate} Hz")
    
    def load_model(self) -> None:
        """Load the Supertonic TTS model."""
        if self.model is not None:
            logger.info("Model already loaded")
            return
        
        logger.info("Loading Supertonic ONNX models...")
        
        try:
            # Model directory
            model_dir = Path.home() / ".cache" / "huggingface" / "supertonic_models" / "onnx"
            voice_style = Path.home() / ".cache" / "huggingface" / "supertonic_models" / "voice_styles" / f"{self.voice_style}.json"
            
            if not model_dir.exists():
                raise FileNotFoundError(
                    f"Supertonic models not found at {model_dir}\n"
                    r"Download with: git clone https://huggingface.co/Supertone/supertonic %USERPROFILE%\.cache\huggingface\supertonic_models"
                )
            
            if not voice_style.exists():
                raise FileNotFoundError(f"Voice style not found: {voice_style}")
            
            # Initialize Supertonic
            self.model = SupertonicTTS(str(model_dir), str(voice_style))
            self.sample_rate = self.model.sample_rate
            
            logger.info(f"[OK] Supertonic loaded successfully")
            logger.info(f"   Sample rate: {self.sample_rate} Hz")
            
        except Exception as e:
            logger.error(f"Failed to load Supertonic: {e}")
            raise RuntimeError(f"Failed to load Supertonic TTS model: {e}")
    
    def synthesize(
        self,
        text: str,
        output_path: Optional[Path] = None
    ) -> np.ndarray:
        """
        Synthesize speech from text.
        
        Args:
            text: Text to convert to speech
            output_path: Optional path to save audio file
            
        Returns:
            Audio waveform as numpy array
        """
        if self.model is None:
            self.load_model()
        
        logger.debug(f"Synthesizing: {text[:50]}...")
        
        try:
            # Generate audio using Supertonic (5 denoising steps, normal speed)
            audio = self.model.synthesize(text, total_steps=5, speed=1.0)
            
            # Ensure float32
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            
            # Save to file if output path provided
            if output_path:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                sf.write(output_path, audio, self.sample_rate)
                logger.debug(f"Audio saved to {output_path}")
            
            logger.debug(f"Generated {len(audio)} samples ({len(audio)/self.sample_rate:.2f}s)")
            return audio
            
        except Exception as e:
            logger.error(f"Failed to synthesize speech: {e}")
            raise RuntimeError(f"Speech synthesis failed: {e}")
    
    @staticmethod
    def get_available_device() -> str:
        """Detect the best available hardware device."""
        return "cpu"  # Supertonic uses CPU-optimized ONNX
    
    def __del__(self):
        """Cleanup when engine is destroyed."""
        if self.model is not None:
            del self.model
