"""
Chatterbox TTS ONNX Wrapper
Uses ONNX models from ResembleAI/chatterbox-turbo-ONNX
Supports GPU/NPU/CPU with dynamic quantization selection
"""

import logging
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import onnxruntime as ort

logger = logging.getLogger(__name__)


class ChatterboxONNX:
    """Chatterbox TTS using ONNX models with hardware optimization"""
    
    def __init__(self, model_dir: str):
        """
        Initialize Chatterbox ONNX TTS
        
        Args:
            model_dir: Path to directory containing ONNX models
        """
        self.model_dir = Path(model_dir)
        self.sample_rate = 24000
        
        # Detect hardware and select quantization
        self.device, self.quant_suffix = self._detect_hardware()
        logger.info(f"Detected hardware: {self.device}, using {self.quant_suffix} quantization")
        
        # ONNX session options
        self.sess_options = ort.SessionOptions()
        self.sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Model sessions (lazy loaded)
        self.language_model = None
        self.speech_encoder = None
        self.conditional_decoder = None
        self.embed_tokens = None
        
    def _detect_hardware(self) -> Tuple[str, str]:
        """
        Detect available hardware and select quantization
        
        Returns:
            (device, quantization_suffix): e.g., ('cuda', 'fp16') or ('cpu', 'q4')
        """
        # Check for CUDA GPU
        available_providers = ort.get_available_providers()
        
        if 'CUDAExecutionProvider' in available_providers:
            logger.info("CUDA GPU detected")
            return 'cuda', 'fp16'
        
        # Check for DirectML (Windows GPU/NPU)
        if 'DmlExecutionProvider' in available_providers:
            logger.info("DirectML (GPU/NPU) detected")
            return 'dml', 'fp16'
        
        # CPU fallback - check system specs
        import psutil
        cpu_count = psutil.cpu_count(logical=False)
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        if cpu_count >= 8 and memory_gb >= 16:
            logger.info("High-end CPU detected, using Q4 quantization")
            return 'cpu', 'q4'
        else:
            logger.info("Standard CPU detected, using full quantization")
            return 'cpu', 'quantized'
    
    def _get_providers(self):
        """Get ONNX execution providers based on hardware"""
        if self.device == 'cuda':
            return ['CUDAExecutionProvider', 'CPUExecutionProvider']
        elif self.device == 'dml':
            return ['DmlExecutionProvider', 'CPUExecutionProvider']
        else:
            return ['CPUExecutionProvider']
    
    def _load_model(self, model_name: str):
        """Load an ONNX model with appropriate quantization"""
        # Construct filename based on quantization
        if self.quant_suffix == 'fp16':
            filename = f"{model_name}_fp16.onnx"
        elif self.quant_suffix == 'q4':
            filename = f"{model_name}_q4.onnx"
        else:
            filename = f"{model_name}_quantized.onnx"
        
        model_path = self.model_dir / filename
        
        # Fallback to base model if specific quantization not found
        if not model_path.exists():
            logger.warning(f"{filename} not found, trying base model")
            model_path = self.model_dir / f"{model_name}.onnx"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        logger.info(f"Loading {filename}...")
        return ort.InferenceSession(
            str(model_path),
            sess_options=self.sess_options,
            providers=self._get_providers()
        )
    
    def _ensure_models(self):
        """Ensure models and their data files are present, download if missing"""
        # Critical files that must exist
        required_files = [
            "language_model.onnx",
            "language_model_q4.onnx",
            "language_model_q4.onnx_data",  # Critical for Q4
            "speech_encoder_q4.onnx_data",
            "conditional_decoder_q4.onnx_data"
        ]
        
        missing = [f for f in required_files if not (self.model_dir / f).exists()]
        
        if not missing:
            return

        logger.info(f"Missing Chatterbox model files: {missing}")
        logger.info("Attempting automatic download via huggingface_hub...")
        
        try:
            from huggingface_hub import snapshot_download
            
            # Download everything in 'onnx' folder
            download_path = snapshot_download(
                repo_id="ResembleAI/chatterbox-turbo-ONNX",
                allow_patterns="onnx/*",  # Get everything in onnx folder
                local_dir=self.model_dir.parent, # Parent of 'onnx'
                local_dir_use_symlinks=False,
                resume_download=True
            )
            logger.info(f"Models downloaded to {download_path}")
            
        except ImportError:
            logger.error("huggingface_hub not installed. Cannot auto-download models.")
            raise ImportError("Please install 'huggingface_hub' or run the setup script.")
        except Exception as e:
            logger.error(f"Auto-download failed: {e}")
            raise RuntimeError(f"Failed to download Chatterbox models: {e}")

    def load_models(self):
        """Load all required ONNX models"""
        if self.language_model is not None:
            return  # Already loaded
        
        self._ensure_models()
        
        logger.info("Loading Chatterbox ONNX models...")
        
        try:
            self.embed_tokens = self._load_model("embed_tokens")
            self.language_model = self._load_model("language_model")
            self.speech_encoder = self._load_model("speech_encoder")
            self.conditional_decoder = self._load_model("conditional_decoder")
            
            logger.info("[OK] All Chatterbox models loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load Chatterbox models: {e}")
            raise RuntimeError(f"Chatterbox model loading failed: {e}")
    
    def synthesize(self, text: str, speed: float = 1.0) -> np.ndarray:
        """
        Synthesize speech from text using autoregressive generation
        """
        if self.language_model is None:
            self.load_models()
        
        try:
            # 1. Prepare Text Embeddings
            text_tokens = self._tokenize(text)
            batch_size, seq_len = text_tokens.shape
            
            # Embed text
            inputs_embeds = self.embed_tokens.run(None, {"input_ids": text_tokens})[0]
            
            # 2. Autoregressive Generation Loop
            # We need to generate speech tokens one by one
            # For this simple implementation, we'll use a fixed length or stop token if known
            # Assuming max generation length proportional to text
            max_new_tokens = min(int(len(text) * 1.5), 300) # Safety limit
            generated_tokens = []
            
            # Initial KV cache (empty)
            num_layers = 24
            num_heads = 16
            head_dim = 64
            
            past_key_values = {}
            for i in range(num_layers):
                past_key_values[f"past_key_values.{i}.key"] = np.zeros(
                    (batch_size, num_heads, 0, head_dim), dtype=np.float32
                )
                past_key_values[f"past_key_values.{i}.value"] = np.zeros(
                    (batch_size, num_heads, 0, head_dim), dtype=np.float32
                )
            
            # Position IDs tracking
            current_pos = 0
            
            # First pass with text
            lm_inputs = {
                "inputs_embeds": inputs_embeds,
                "attention_mask": np.ones((batch_size, seq_len), dtype=np.int64),
                "position_ids": np.arange(seq_len, dtype=np.int64).reshape(1, -1),
                **past_key_values
            }
            
            # Run LM first pass
            outputs = self.language_model.run(None, lm_inputs)
            logits = outputs[0]
            
            # Update KV cache from model output
            # Output structure: [logits, present.0.key, present.0.value, ..., present.23.value]
            # present keys start at index 1
            for i in range(num_layers):
                past_key_values[f"past_key_values.{i}.key"] = outputs[1 + i*2]
                past_key_values[f"past_key_values.{i}.value"] = outputs[2 + i*2]
            
            # Greedy decode first token
            next_token_id = np.argmax(logits[:, -1, :], axis=-1).reshape(1, 1)
            generated_tokens.append(next_token_id.item())
            current_pos += seq_len
            
            # Generation Loop
            for _ in range(max_new_tokens):
                # Embed next token
                next_embeds = self.embed_tokens.run(None, {"input_ids": next_token_id.astype(np.int64)})[0]
                
                # Prepare inputs for single step
                step_inputs = {
                    "inputs_embeds": next_embeds,
                    "attention_mask": np.ones((batch_size, current_pos + 1), dtype=np.int64), # Global mask? Or just 1?
                    # Note: Attention mask shape usually needs to match total sequence length including float, 
                    # but for caching inference, many models accept mask for just the new step with full history?
                    # Let's try passing mask of 1s with shape (batch, 1) + past length?
                    # Actually standard HF ONNX export expects mask for total length
                    "attention_mask": np.ones((batch_size, current_pos + 1), dtype=np.int64),
                    "position_ids": np.array([[current_pos]], dtype=np.int64),
                    **past_key_values
                }
                
                outputs = self.language_model.run(None, step_inputs)
                logits = outputs[0]
                
                # Update KVs
                for i in range(num_layers):
                    past_key_values[f"past_key_values.{i}.key"] = outputs[1 + i*2]
                    past_key_values[f"past_key_values.{i}.value"] = outputs[2 + i*2]
                
                next_token_id = np.argmax(logits[:, -1, :], axis=-1).reshape(1, 1)
                
                # Check for EOS (assuming 0 or specific token, but unknown, so just run max len)
                # if next_token_id.item() == EOS: break
                
                generated_tokens.append(next_token_id.item())
                current_pos += 1
            
            # 3. Decode to Audio
            
            # Use speech_encoder to get valid default embeddings from dummy audio
            # Create dummy audio (0.5s of silence)
            dummy_wav = np.zeros((1, 16000), dtype=np.float32)
            
            # Ensure speech_encoder is loaded
            if self.speech_encoder is None:
                 self.load_models()
                 
            encoder_inputs = {
                "audio_values": dummy_wav
            }
            
            # Run encoder
            # Output structure (from log):
            # 0: audio_features [B, Seq, 1024]
            # 1: audio_tokens [B, Seq]
            # 2: speaker_embeddings [B, 192]
            # 3: speaker_features [B, Seq, 80]
            enc_outputs = self.speech_encoder.run(None, encoder_inputs)
            speaker_embeddings = enc_outputs[2] # [Batch, 192]
            
            # Ensure 2D rank (Batch, Dim)
            if speaker_embeddings.ndim == 3:
                if speaker_embeddings.shape[1] == 1:
                    speaker_embeddings = speaker_embeddings.squeeze(1)
                else:
                    # If multiple frames, take mean as global embedding
                    speaker_embeddings = speaker_embeddings.mean(axis=1)
                
            speaker_features = enc_outputs[3]   # [Batch, Seq, 80]
            
            # Prepare speech tokens input
            speech_tokens = np.array([generated_tokens], dtype=np.int64)
            
            decoder_inputs = {
                "speech_tokens": speech_tokens,
                "speaker_embeddings": speaker_embeddings,
                "speaker_features": speaker_features
            }
            
            # Check if audio_features is required?
            # From previous inspection: speech_tokens, speaker_embeddings were main inputs
            # 'audio_features' was seen in dump? Let's assume it handles missing prompts
            
            audio_output = self.conditional_decoder.run(None, decoder_inputs)[0]
            
            # Ensure float32 output
            if audio_output.dtype != np.float32:
                audio_output = audio_output.astype(np.float32)
            
            # Flatten
            if audio_output.ndim > 1:
                audio_output = audio_output.flatten()
            
            return audio_output
            
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            raise RuntimeError(f"Chatterbox synthesis error: {e}")
    
    def _tokenize(self, text: str) -> np.ndarray:
        """
        Simple character-level tokenization
        TODO: Replace with proper tokenizer when available
        """
        # Convert to ASCII codes as a simple tokenizer
        tokens = np.array([ord(c) for c in text], dtype=np.int64)
        return tokens.reshape(1, -1)  # Add batch dimension
