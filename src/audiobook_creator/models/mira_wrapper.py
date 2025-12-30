import os
import torch
import numpy as np
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from .ncodec.codec import TTSCodec

logger = logging.getLogger(__name__)

class MiraWrapper:
    """
    Wrapper for MiraTTS (FastNeuTTS)
    Uses transformers for LLM and vendored NCodec for audio decoding
    """
    
    def __init__(self, device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load LLM
        logger.info(f"Loading MiraTTS on {self.device}...")
        self.dtype = torch.bfloat16 if self.device == 'cuda' else torch.float32
        
        model_id = "YatharthS/MiraTTS"
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=self.dtype,
            device_map=self.device,
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model.eval()
        
        # Initialize Codec
        logger.info("Initializing NCodec...")
        # Codec usually loads its own models (Processer, Encoder, Quantizer, Decoder)
        # NCodec by default downloads models from HF if not present
        self.codec = TTSCodec() 
        
        # Generation Config
        self.gen_config = GenerationConfig(
            top_p=0.95,
            top_k=50,
            temperature=0.8,
            max_new_tokens=1024,
            repetition_penalty=1.2,
            do_sample=True,
            min_p=0.05
        )

    def infer(self, text: str, ref_audio_path: str = None) -> np.ndarray:
        """
        Generate audio from text
        Args:
            text: Text to speak
            ref_audio_path: Optional path to reference audio for voice cloning
                            If None, uses a default or random voice (NCodec handles this)
        """
        
        # 1. Encode Reference Audio (Context Tokens)
        # If no ref audio, NCodec might need a way to get default context
        # Looking at MiraTTS code: context_tokens = codec.encode(audio_file)
        
        # If ref_audio_path is None, we need a default behavior.
        # NCodec might not have a "random" generator built-in easily accessible here?
        # Let's assume for now we might need a dummy file or we handle it if NCodec supports it.
        # For this V1 implementation, we will try to rely on provided ref audio 
        # OR fail gracefully if none provided (user needs to select one).
        
        context_tokens = None
        if ref_audio_path:
             context_tokens = self.codec.encode(ref_audio_path)
        else:
            # Generate a "dummy" context to avoid failing
            # NCodec encode takes a path, so we'll create a temporary silence file if needed
            # Or better, we can mock the encoding if we know what it produces.
            # But safer to just give it silence.
            import tempfile
            import soundfile as sf
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                dummy_wav = np.zeros(16000 * 5, dtype=np.float32) # 5s of silence
                sf.write(tf.name, dummy_wav, 16000)
                temp_path = tf.name
            
            try:
                context_tokens = self.codec.encode(temp_path)
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

        # 2. Format Prompt
        # Using codec to format prompt
        formatted_prompt = self.codec.format_prompt(text, context_tokens, None)
        
        # 3. Generate (LLM)
        inputs = self.tokenizer(formatted_prompt, return_tensors='pt').to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=self.gen_config
            )
            
        # Decode output
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # The output of the LLM includes the prompt + generated audio tokens (as text/pseudo-tokens)
        # We need to extract the "response" part or pass the whole thing to codec.decode?
        # MiraTTS original code: audio = self.codec.decode(response[0].text, context_tokens)
        # where response.text is the generated text.
        
        # We need to be careful: model.generate returns input+output. 
        # We should slice off the input?
        # Actually tokenizer.decode usually decodes the whole sequence. 
        # Let's check if codec.decode expects the full string or just the new part.
        # Usually these things expect the full generated string which contains special tokens.
        
        # Extract the *new* part?
        # input_length = inputs.input_ids.shape[1]
        # visible_tokens = outputs[0][input_length:]
        # generated_text_only = self.tokenizer.decode(visible_tokens, skip_special_tokens=False) 
        
        # Actually, let's trust that NCodec's `decode` can parse the output string.
        # But we must be careful about `skip_special_tokens`. 
        # Audio tokens might be special tokens. So skip_special_tokens=False is safer?
        # Wait, if they are semantic tokens mapped to text, they are normal tokens. 
        # If they are special added tokens, we need False.
        # Safer to use False.
        
        full_output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        # 4. Decode Audio
        audio = self.codec.decode(full_output_text, context_tokens)
        
        # Audio is [1, Samples] torch tensor?
        if isinstance(audio, torch.Tensor):
            audio = audio.squeeze().cpu().numpy()
            
        return audio
