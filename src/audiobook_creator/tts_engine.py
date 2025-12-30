"""
TTS Engine Module - Modular Architecture

Supports multiple TTS backends:
- Supertone Supertonic (ONNX)
- Kokoro (Open Weight)
- ResembleAI Chatterbox (Transformers)
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
import soundfile as sf

from .supertonic_wrapper import SupertonicTTS


logger = logging.getLogger(__name__)


class TTSEngine(ABC):
    """Abstract base class for TTS engines."""
    
    def __init__(self, device: str = "cpu", voice_style: str = "default"):
        self.device = device
        self.voice_style = voice_style
        self.sample_rate = 24000  # Default, should be overridden
        self.model = None
        self.max_text_length = 500  # Default limit
    
    @abstractmethod
    def load_model(self) -> None:
        """Load the TTS model."""
        pass
    
    @abstractmethod
    def _synthesize_single(self, text: str) -> np.ndarray:
        """Synthesize a single chunk of text (internal)."""
        pass

    def synthesize(self, text: str, output_path: Optional[Path] = None) -> np.ndarray:
        """
        Synthesize speech from text, automatically splitting if too long or contains newlines.
        """
        if self.model is None:
            self.load_model()
            
        # Always split on newlines first for natural pauses
        if '\n' in text or len(text) > self.max_text_length:
            audio = self._split_and_synthesize(text)
        else:
            audio = self._synthesize_single(text)
            
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(output_path, audio, self.sample_rate)
            
        return audio

    def _split_and_synthesize(self, text: str) -> np.ndarray:
        """Split text into chunks and synthesize each."""
        import re
        
        # 1. Detect and split on line breaks for natural pauses
        if '\n' in text:
            lines = text.split('\n')
            audio_segments = []
            for line in lines:
                line = line.strip()
                if not line: continue
                audio_segments.append(self.synthesize(line))
                # 200ms pause
                audio_segments.append(np.zeros(int(self.sample_rate * 0.2), dtype=np.float32))
            return np.concatenate(audio_segments[:-1]) if audio_segments else np.zeros(0)

        # 2. Split by sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # If there's only one "sentence" but it's too long, split by punctuation or space
        if len(sentences) == 1 and len(sentences[0]) > self.max_text_length:
            sentences = re.split(r'(?<=[,;])\s+|\s+', text)
            
        audio_segments = []
        current_chunk = ""
        
        for part in sentences:
            # If a single part is STILL too long, split it by character count
            if len(part) > self.max_text_length:
                if current_chunk:
                    audio_segments.append(self._synthesize_single(current_chunk))
                    audio_segments.append(np.zeros(int(self.sample_rate * 0.1), dtype=np.float32))
                    current_chunk = ""
                
                for i in range(0, len(part), self.max_text_length):
                    sub_part = part[i:i + self.max_text_length]
                    audio_segments.append(self._synthesize_single(sub_part))
                    if i + self.max_text_length < len(part):
                        audio_segments.append(np.zeros(int(self.sample_rate * 0.05), dtype=np.float32))
                continue

            if len(current_chunk) + len(part) + 1 < self.max_text_length:
                current_chunk += " " + part if current_chunk else part
            else:
                if current_chunk:
                    audio_segments.append(self._synthesize_single(current_chunk))
                    audio_segments.append(np.zeros(int(self.sample_rate * 0.1), dtype=np.float32))
                current_chunk = part
        
        if current_chunk:
            audio_segments.append(self._synthesize_single(current_chunk))
            
        return np.concatenate(audio_segments) if audio_segments else np.zeros(0)

    @staticmethod
    def get_available_device() -> str:
        """Detect the best available hardware device."""
        from .hardware import select_best_device
        type, name = select_best_device()
        return name # Returns e.g. "cuda:0", "NPU", "cpu"


class SupertonicEngine(TTSEngine):
    """Supertone Supertonic Engine (ONNX)."""
    
    def __init__(self, device: str = "cpu", voice_style: str = "M1"):
        super().__init__(device, voice_style)
        self.sample_rate = 44100
        self.max_text_length = 300  # Strict limit for Supertonic
        
        logger.info("Initializing Supertonic TTS engine")
        logger.info(f"Voice style: {voice_style}")
        logger.info("Description: Supertone Supertonic - ONNX based, ultra-fast")
    
    def load_model(self) -> None:
        if self.model is not None:
            return
            
        logger.info("Loading Supertonic ONNX models...")
        try:
            # Model directory
            model_dir = Path.home() / ".cache" / "huggingface" / "supertonic_models" / "onnx"
            voice_style_path = Path.home() / ".cache" / "huggingface" / "supertonic_models" / "voice_styles" / f"{self.voice_style}.json"
            
            if not model_dir.exists():
                raise FileNotFoundError(f"Supertonic models not found at {model_dir}")
            
            if not voice_style_path.exists():
                raise FileNotFoundError(f"Voice style not found: {voice_style_path}")
            
            self.model = SupertonicTTS(str(model_dir), str(voice_style_path))
            self.sample_rate = self.model.sample_rate
            logger.info(f"[OK] Supertonic loaded successfully (SR: {self.sample_rate} Hz)")
            
        except Exception as e:
            logger.error(f"Failed to load Supertonic: {e}")
            raise RuntimeError(f"Failed to load Supertonic TTS model: {e}")

    def _synthesize_single(self, text: str) -> np.ndarray:
        try:
            audio = self.model.synthesize(text, total_steps=5, speed=1.0)
            
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
                
            return audio
        except Exception as e:
            logger.error(f"Failed to synthesize speech: {e}")
            raise RuntimeError(f"Speech synthesis failed: {e}")


class KokoroEngine(TTSEngine):
    """Kokoro TTS Engine (Open Weight)."""
    
    def __init__(self, device: str = "cpu", voice_style: str = "af"):
        super().__init__(device, voice_style)
        self.sample_rate = 24000
        self.max_text_length = 500  # Kokoro limit
        logger.info("Initializing Kokoro TTS engine")
        logger.info(f"Voice style: {voice_style}")
        logger.info("Description: Kokoro - Lightweight, high-quality open weight model")

    def load_model(self) -> None:
        if self.model is not None:
            return
            
        logger.info("Loading Kokoro model...")
        try:
            from kokoro import KPipeline
            # Initialize pipeline (lang_code='a' for American English)
            self.model = KPipeline(lang_code='a')
            logger.info("[OK] Kokoro loaded successfully")
        except ImportError:
            logger.error("Kokoro package not found. Install with: pip install kokoro>=0.3.4 soundfile")
            raise ImportError("Kokoro package not found")
        except Exception as e:
            logger.error(f"Failed to load Kokoro: {e}")
            raise RuntimeError(f"Failed to load Kokoro TTS model: {e}")

    def _synthesize_single(self, text: str) -> np.ndarray:
        try:
            # Generate audio
            # generator returns (graphemes, phonemes, audio)
            generator = self.model(text, voice=self.voice_style, speed=1.0)
            
            # Concatenate all audio chunks
            audio_chunks = []
            for _, _, audio in generator:
                audio_chunks.append(audio)
            
            if not audio_chunks:
                return np.zeros(0, dtype=np.float32)
                
            return np.concatenate(audio_chunks)
        except Exception as e:
            logger.error(f"Failed to synthesize speech with Kokoro: {e}")
            raise RuntimeError(f"Kokoro synthesis failed: {e}")


class ChatterboxEngine(TTSEngine):
    """Chatterbox TTS - ONNX based, hardware optimized"""
    
    def __init__(self, device: str = "cpu", voice_style: str = "default"):
        super().__init__(device, voice_style)
        self.sample_rate = 24000
        self.max_text_length = 500  # Chatterbox limit
        logger.info("Initializing Chatterbox TTS engine (ONNX)")
        logger.info("Description: ResembleAI Chatterbox-Turbo - Expressive, hardware-optimized")

    def load_model(self) -> None:
        if self.model is not None:
            return
            
        logger.info("Loading Chatterbox ONNX models...")
        try:
            from .chatterbox_wrapper import ChatterboxONNX
            
            model_dir = Path.home() / ".cache" / "huggingface" / "chatterbox_models" / "onnx"
            
            if not model_dir.exists():
                raise FileNotFoundError(
                    f"Chatterbox models not found at {model_dir}\n"
                    f"Download from: https://huggingface.co/ResembleAI/chatterbox-turbo-ONNX/tree/main/onnx"
                )
            
            self.model = ChatterboxONNX(str(model_dir))
            self.model.load_models()
            self.sample_rate = self.model.sample_rate
            
            logger.info("[OK] Chatterbox loaded successfully")
        except ImportError as e:
            logger.error(f"Chatterbox wrapper not found: {e}")
            raise ImportError("Chatterbox ONNX wrapper not available")
        except Exception as e:
            logger.error(f"Failed to load Chatterbox: {e}")
            raise RuntimeError(f"Failed to load Chatterbox TTS model: {e}")

    def _synthesize_single(self, text: str) -> np.ndarray:
        try:
            return self.model.synthesize(text, speed=1.0)
        except Exception as e:
            logger.error(f"Failed to synthesize speech with Chatterbox: {e}")
            raise RuntimeError(f"Chatterbox synthesis failed: {e}")



class SopranoEngine(TTSEngine):
    """Soprano-80M TTS Engine."""
    
    def __init__(self, device: str = "cpu", voice_style: str = "default"):
        super().__init__(device, voice_style)
        self.sample_rate = 32000
        self.max_text_length = 400
        logger.info("Initializing Soprano TTS engine")
        logger.info("Description: Soprano-80M - Ultra-lightweight with Vocos decoder")

    def load_model(self) -> None:
        if self.model is not None:
            return
        
        logger.info("Loading Soprano-80M...")
        try:
            from .models.soprano_wrapper import SopranoWrapper
            self.model = SopranoWrapper(device=self.device)
            # Soprano sample rate is 32000
            self.sample_rate = 32000
            logger.info("[OK] Soprano loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Soprano: {e}")
            raise RuntimeError(f"Failed to load Soprano TTS model: {e}")

    def _synthesize_single(self, text: str) -> np.ndarray:
        try:
            return self.model.infer(text)
        except Exception as e:
            logger.error(f"Soprano synthesis failed: {e}")
            raise RuntimeError(f"Soprano synthesis failed: {e}")


class MiraEngine(TTSEngine):
    """MiraTTS Engine."""
    
    def __init__(self, device: str = "cpu", voice_style: str = "default"):
        super().__init__(device, voice_style)
        self.sample_rate = 48000
        self.max_text_length = 400
        logger.info("Initializing MiraTTS engine")
        logger.info("Description: MiraTTS - High quality 48kHz, NCodec based")

    def load_model(self) -> None:
        if self.model is not None:
            return
        
        logger.info("Loading MiraTTS...")
        try:
            from .models.mira_wrapper import MiraWrapper
            # Basic device check, ignoring device arg if gpu available
            device = 'cuda' if self.device == 'cuda' or (self.device == 'auto' and torch.cuda.is_available()) else 'cpu'
            self.model = MiraWrapper(device=device)
            self.sample_rate = 48000
            logger.info("[OK] MiraTTS loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load MiraTTS: {e}")
            raise RuntimeError(f"Failed to load MiraTTS model: {e}")

    def _synthesize_single(self, text: str) -> np.ndarray:
        try:
            # TODO: Add voice cloning support via voice_style (path to audio)
            ref_audio = None
            if self.voice_style and self.voice_style != "default":
                 # If voice_style is a path, use it
                 p = Path(self.voice_style)
                 if p.exists():
                     ref_audio = str(p)
            
            return self.model.infer(text, ref_audio_path=ref_audio)
        except Exception as e:
            logger.error(f"MiraTTS synthesis failed: {e}")
            raise RuntimeError(f"MiraTTS synthesis failed: {e}")


def get_tts_engine(model_name: str, device: str = "cpu", voice_style: str = "default") -> TTSEngine:
    """Factory function to get the appropriate TTS engine."""
    model_name = model_name.lower()
    
    if model_name == "supertonic":
        style = "M1" if voice_style == "default" else voice_style
        return SupertonicEngine(device, style)
    
    elif model_name == "kokoro":
        style = "af" if voice_style == "default" else voice_style
        return KokoroEngine(device, style)
        
    elif model_name == "chatterbox":
        return ChatterboxEngine(device, voice_style)
        
    elif model_name == "soprano":
        return SopranoEngine(device, voice_style)
        
    elif model_name == "miratts":
        return MiraEngine(device, voice_style)
        
    else:
        raise ValueError(f"Unknown TTS model: {model_name}")
