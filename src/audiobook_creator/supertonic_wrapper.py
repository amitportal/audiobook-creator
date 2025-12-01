"""
Supertone TTS Wrapper (based on official implementation)

Simplified version for audiobook-creator using ONNX Runtime
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Optional, Tuple
from unicodedata import normalize

import numpy as np
import onnxruntime as ort


logger = logging.getLogger(__name__)


class UnicodeProcessor:
    """Process text into unicode indices for Supertonic"""
    
    def __init__(self, unicode_indexer_path: str):
        with open(unicode_indexer_path, 'r') as f:
            self.indexer = json.load(f)
    
    def _preprocess_text(self, text: str) -> str:
        """Clean and normalize text"""
        text = normalize("NFKD", text)
        
        # Remove emojis
        emoji_pattern = re.compile(
            "[\U0001f600-\U0001f64f"  # emoticons
            "\U0001f300-\U0001f5ff"  # symbols & pictographs
            "\U0001f680-\U0001f6ff"  # transport & map symbols
            "\U0001f700-\U0001f77f"
            "\U0001f780-\U0001f7ff"
            "\U0001f800-\U0001f8ff"
            "\U0001f900-\U0001f9ff"
            "\U0001fa00-\U0001fa6f"
            "\U0001fa70-\U0001faff"
            "\u2600-\u26ff"
            "\u2700-\u27bf"
            "\U0001f1e6-\U0001f1ff]+",
            flags=re.UNICODE
        )
        text = emoji_pattern.sub("", text)
        
        # Character replacements
        replacements = {
            "–": "-", "‑": "-", "—": "-", "¯": " ", "_": " ",
            """: '"', """: '"', "'": "'", "'": "'", "´": "'", "`": "'",
            "[": " ", "]": " ", "|": " ", "/": " ", "#": " ",
            "→": " ", "←": " "
        }
        for k, v in replacements.items():
            text = text.replace(k, v)
        
        # Remove combining diacritics
        text = re.sub(
            r"[\u0302\u0303\u0304\u0305\u0306\u0307\u0308\u030A\u030B\u030C\u0327\u0328\u0329\u032A\u032B\u032C\u032D\u032E\u032F]",
            "", text
        )
        
        # Expression replacements
        text = text.replace("@", " at ")
        text = text.replace("e.g.,", "for example, ")
        text = text.replace("i.e.,", "that is, ")
        
        # Fix spacing
        text = re.sub(r" ,", ",", text)
        text = re.sub(r" \.", ".", text)
        text = re.sub(r" !", "!", text)
        text = re.sub(r" \?", "?", text)
        text = re.sub(r" ;", ";", text)
        text = re.sub(r" :", ":", text)
        
        # Remove extra spaces
        text = re.sub(r"\s+", " ", text).strip()
        
        # Add period if needed
        if not re.search(r"[.!?;:,'\"')\]}…。」』】〉》›»]$", text):
            text += "."
        
        return text
    
    def __call__(self, text_list: list) -> Tuple[np.ndarray, np.ndarray]:
        """Convert list of texts to unicode indices and masks"""
        text_list = [self._preprocess_text(t) for t in text_list]
        text_ids_lengths = np.array([len(text) for text in text_list], dtype=np.int64)
        text_ids = np.zeros((len(text_list), text_ids_lengths.max()), dtype=np.int64)
        
        for i, text in enumerate(text_list):
            # Get unicode values and look up their IDs in the indexer (which is a list)
            unicode_vals = [ord(char) for char in text]
            text_ids[i, :len(unicode_vals)] = np.array(
                [self.indexer[val] for val in unicode_vals], dtype=np.int64
            )
        
        # Create mask
        max_len = text_ids_lengths.max()
        ids = np.arange(0, max_len)
        mask = (ids < np.expand_dims(text_ids_lengths, axis=1)).astype(np.float32)
        text_mask = mask.reshape(-1, 1, max_len)
        
        return text_ids, text_mask


class SupertonicTTS:
    """Supertonic Text-to-Speech Engine"""
    
    def __init__(self, model_dir: str, voice_style_path: str):
        """
        Initialize Supertonic TTS
        
        Args:
            model_dir: Path to directory containing ONNX models
            voice_style_path: Path to voice style JSON file (e.g., M1.json)
        """
        self.model_dir = Path(model_dir)
        
        # Load configuration
        cfg_path = self.model_dir / "tts.json"
        with open(cfg_path, 'r') as f:
            self.cfgs = json.load(f)
        
        self.sample_rate = self.cfgs["ae"]["sample_rate"]
        self.base_chunk_size = self.cfgs["ae"]["base_chunk_size"]
        self.chunk_compress_factor = self.cfgs["ttl"]["chunk_compress_factor"]
        self.ldim = self.cfgs["ttl"]["latent_dim"]
        
        # Load ONNX models
        logger.info("Loading Supertonic ONNX models...")
        opts = ort.SessionOptions()
        providers = ["CPUExecutionProvider"]
        
        self.dp_ort = ort.InferenceSession(
            str(self.model_dir / "duration_predictor.onnx"),
            sess_options=opts, providers=providers
        )
        self.text_enc_ort = ort.InferenceSession(
            str(self.model_dir / "text_encoder.onnx"),
            sess_options=opts, providers=providers
        )
        self.vector_est_ort = ort.InferenceSession(
            str(self.model_dir / "vector_estimator.onnx"),
            sess_options=opts, providers=providers
        )
        self.vocoder_ort = ort.InferenceSession(
            str(self.model_dir / "vocoder.onnx"),
            sess_options=opts, providers=providers
        )
        
        # Load text processor
        unicode_indexer_path = self.model_dir / "unicode_indexer.json"
        self.text_processor = UnicodeProcessor(str(unicode_indexer_path))
        
        # Load voice style
        self._load_voice_style(voice_style_path)
        
        logger.info("[OK] Supertonic TTS initialized successfully")
    
    def _load_voice_style(self, voice_style_path: str):
        """Load voice style from JSON"""
        with open(voice_style_path, 'r') as f:
            voice_style = json.load(f)
        
        ttl_dims = voice_style["style_ttl"]["dims"]
        dp_dims = voice_style["style_dp"]["dims"]
        
        ttl_data = np.array(voice_style["style_ttl"]["data"], dtype=np.float32).flatten()
        self.style_ttl = ttl_data.reshape(1, ttl_dims[1], ttl_dims[2])
        
        dp_data = np.array(voice_style["style_dp"]["data"], dtype=np.float32).flatten()
        self.style_dp = dp_data.reshape(1, dp_dims[1], dp_dims[2])
    
    def synthesize(
        self,
        text: str,
        total_steps: int = 5,
        speed: float = 1.05
    ) -> np.ndarray:
        """
        Synthesize speech from text
        
        Args:
            text: Input text
            total_steps: Number of denoising steps (default: 5)
            speed: Speech speed (higher = faster, default: 1.05)
            
        Returns:
            Audio waveform as numpy array
        """
        # Limit text length to avoid ONNX dimension errors
        # Supertonic has max text length of ~300 characters per call
        MAX_TEXT_LENGTH = 300
        
        # First, detect and split on line breaks for natural pauses
        if '\n' in text:
            lines = text.split('\n')
            audio_segments = []
            
            for line in lines:
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                
                # Synthesize each line
                line_audio = self.synthesize(line, total_steps, speed)  # Recursive call
                audio_segments.append(line_audio)
                
                # Add 200ms pause after each line break
                pause = np.zeros(int(self.sample_rate * 0.2), dtype=np.float32)
                audio_segments.append(pause)
            
            # Remove final pause
            if audio_segments and len(audio_segments) > 1:
                audio_segments = audio_segments[:-1]
            
            # Concatenate all segments
            if audio_segments:
                return np.concatenate(audio_segments)
            else:
                return np.zeros(int(self.sample_rate * 0.1), dtype=np.float32)
        
        if len(text) > MAX_TEXT_LENGTH:
            # Split into smaller chunks and concatenate audio
            import re
            
            # Split by sentences while respecting max length
            sentences = re.split(r'(?<=[.!?])\s+', text)
            audio_segments = []
            current_chunk = ""
            
            for sentence in sentences:
                if len(current_chunk) + len(sentence) < MAX_TEXT_LENGTH:
                    current_chunk += " " + sentence if current_chunk else sentence
                else:
                    # Process current chunk
                    if current_chunk:
                        chunk_audio = self._synthesize_single(current_chunk, total_steps, speed)
                        audio_segments.append(chunk_audio)
                        # Add small pause between chunks (100ms)
                        pause = np.zeros(int(self.sample_rate * 0.1), dtype=np.float32)
                        audio_segments.append(pause)
                    
                    # Start new chunk with current sentence
                    current_chunk = sentence
            
            # Process remaining chunk
            if current_chunk:
                chunk_audio = self._synthesize_single(current_chunk, total_steps, speed)
                audio_segments.append(chunk_audio)
            
            # Concatenate all segments
            if audio_segments:
                return np.concatenate(audio_segments)
            else:
                return np.zeros(int(self.sample_rate * 0.1), dtype=np.float32)
        else:
            return self._synthesize_single(text, total_steps, speed)
    
    def _synthesize_single(
        self,
        text: str,
        total_steps: int = 5,
        speed: float = 1.05
    ) -> np.ndarray:
        """
        Synthesize speech from a single text chunk (internal method)
        
        Args:
            text: Input text (should be < 300 chars)
            total_steps: Number of denoising steps
            speed: Speech speed
            
        Returns:
            Audio waveform as numpy array
        """
        # Process text
        text_ids, text_mask = self.text_processor([text])
        
        # Predict duration
        dur_onnx, = self.dp_ort.run(
            None,
            {"text_ids": text_ids, "style_dp": self.style_dp, "text_mask": text_mask}
        )
        dur_onnx = dur_onnx / speed
        
        # Encode text
        text_emb_onnx, = self.text_enc_ort.run(
            None,
            {"text_ids": text_ids, "style_ttl": self.style_ttl, "text_mask": text_mask}
        )
        
        # Sample noisy latent
        xt, latent_mask = self._sample_noisy_latent(dur_onnx)
        
        # Denoise (vector estimation)
        total_step_np = np.array([total_steps], dtype=np.float32)
        for step in range(total_steps):
            current_step = np.array([step], dtype=np.float32)
            xt, = self.vector_est_ort.run(
                None,
                {
                    "noisy_latent": xt,
                    "text_emb": text_emb_onnx,
                    "style_ttl": self.style_ttl,
                    "text_mask": text_mask,
                    "latent_mask": latent_mask,
                    "current_step": current_step,
                    "total_step": total_step_np
                }
            )
        
        # Vocoder
        wav, = self.vocoder_ort.run(None, {"latent": xt})
        
        # Trim to actual duration
        duration_samples = int(self.sample_rate * dur_onnx[0])
        wav = wav[0, :duration_samples]
        
        return wav.astype(np.float32)
    
    def _sample_noisy_latent(
        self,
        duration: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sample initial noisy latent for denoising"""
        bsz = len(duration)
        wav_len_max = duration.max() * self.sample_rate
        wav_lengths = (duration * self.sample_rate).astype(np.int64)
        chunk_size = self.base_chunk_size * self.chunk_compress_factor
        latent_len = int((wav_len_max + chunk_size - 1) / chunk_size)
        latent_dim = self.ldim * self.chunk_compress_factor
        
        noisy_latent = np.random.randn(bsz, latent_dim, latent_len).astype(np.float32)
        
        # Create latent mask
        latent_size = self.base_chunk_size * self.chunk_compress_factor
        latent_lengths = (wav_lengths + latent_size - 1) // latent_size
        max_len = latent_lengths.max()
        ids = np.arange(0, max_len)
        mask = (ids < np.expand_dims(latent_lengths, axis=1)).astype(np.float32)
        latent_mask = mask.reshape(-1, 1, max_len)
        
        noisy_latent = noisy_latent * latent_mask
        
        return noisy_latent, latent_mask
