import re
import torch
import numpy as np
import onnxruntime as ort
from ...fastaudiosr import FASR
from .model_utils import AudioTokenizer

class AudioDecoder:
    def __init__(self, decoder_paths):
        
        from ....hardware import select_best_device, get_onnx_providers
        hw_type, hw_name = select_best_device()
        self.device = hw_name
        providers = get_onnx_providers(hw_type)
            
        self.processor_detokenizer = ort.InferenceSession(f"{decoder_paths}/processer.onnx", sess_options, providers=providers)
        self.audio_detokenizer = AudioTokenizer(f'{decoder_paths}/detokenizer.safetensors')
        self.upsampler = FASR(f'{decoder_paths}/upsampler.pth', device=self.device)
        if "cuda" in self.device:
            _ = self.upsampler.model.half()
        
    @torch.inference_mode()    
    def detokenize(self, context_tokens, speech_tokens):
        """helper function to detokenize"""
        
        speech_tokens = (
            torch.tensor([int(token) for token in re.findall(r"speech_token_(\d+)", speech_tokens)])
            .long()
            .unsqueeze(0)
        ).numpy()
        context_tokens = (
            torch.tensor([int(token) for token in re.findall(r"context_token_(\d+)", context_tokens)])
            .long()
            .unsqueeze(0).unsqueeze(0)
        ).numpy().astype(np.int32)
        
        x = self.processor_detokenizer.run(["preprocessed_output"], {"context_tokens": context_tokens, "speech_tokens": speech_tokens})
        x = torch.from_numpy(x[0]).to(self.device)
        lowres_wav = self.audio_detokenizer.decode(x).squeeze(0)
        if "cuda" in self.device:
            lowres_wav = lowres_wav.half()
        u_wav = self.upsampler.run(lowres_wav)
        return u_wav
