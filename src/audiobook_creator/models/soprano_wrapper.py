
import torch
import numpy as np
import logging
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download
from .vocos.decoder import SopranoDecoder

logger = logging.getLogger(__name__)

class SopranoWrapper:
    """
    Wrapper for Soprano-80M TTS model
    Uses transformers for LLM and vendored Vocos for decoding
    """
    
    def __init__(self, device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load LLM
        logger.info(f"Loading Soprano-80M on {self.device}...")
        self.dtype = torch.bfloat16 if self.device == 'cuda' else torch.float32
        
        self.model = AutoModelForCausalLM.from_pretrained(
            'ekwek/Soprano-80M',
            dtype=self.dtype,
            device_map=self.device,
            trust_remote_code=True 
        )
        self.tokenizer = AutoTokenizer.from_pretrained('ekwek/Soprano-80M')
        self.model.eval()
        
        # Load Decoder (Vocos)
        logger.info("Loading Soprano Decoder...")
        self.decoder = SopranoDecoder().to(self.device).to(self.dtype)
        
        # Download decoder weights
        decoder_path = hf_hub_download(repo_id='ekwek/Soprano-80M', filename='decoder.pth')
        self.decoder.load_state_dict(torch.load(decoder_path, map_location=self.device))
        self.decoder.eval()
        
        # Constants
        self.RECEPTIVE_FIELD = 4
        self.TOKEN_SIZE = 2048
        
    def infer(self, text: str) -> np.ndarray:
        """Generating audio from text"""
        
        # 1. Prepare Text
        # Soprano expects specific formatting
        formatted_text = f"[STOP][TEXT]{text.strip()}[START]"
        
        inputs = self.tokenizer(
            [formatted_text],
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # 2. Generate Hidden States (LLM)
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=512,
                do_sample=True,
                top_p=0.95,
                temperature=0.3, # Low temp for TTS stability
                repetition_penalty=1.2,
                pad_token_id=self.tokenizer.pad_token_id,
                return_dict_in_generate=True,
                output_hidden_states=True
            )
            
        # Extract last hidden states from generation
        # Soprano uses the last hidden state of each generated token
        hidden_states = []
        eos_token_id = self.model.config.eos_token_id
        seq = outputs.sequences[0]
        
        num_output_tokens = len(outputs.hidden_states)
        prompt_len = inputs['input_ids'].size(1)
        
        for j in range(num_output_tokens):
            # Safe indexing
            token = seq[prompt_len + j]
            if token != eos_token_id:
                # Get last layer hidden state
                hidden_states.append(outputs.hidden_states[j][-1][0, -1, :])
        
        if not hidden_states:
            return np.zeros(0, dtype=np.float32)
            
        # Stack hidden states: [SeqLen, HiddenDim]
        last_hidden_state = torch.stack(hidden_states).to(self.dtype)
        
        # 3. Decode to Audio (Vocos)
        # Decoder expects: [Batch, Hidden, Seq]
        # We need to reshape/transpose
        
        # Prepare input for decoder (Batch=1)
        # Decoder input shape: [1, 512, SeqLen] (assuming hidden dim 512)
        # Validating dim... Soprano hidden size is 512
        
        decoder_input = last_hidden_state.unsqueeze(0).transpose(1, 2)
        
        with torch.no_grad():
            audio = self.decoder(decoder_input)
            
        # Audio is [1, Samples]
        return audio.squeeze().cpu().float().numpy()
