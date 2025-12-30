import gc
import torch
import librosa
import numpy as np
from .decoder.model import AudioDecoder
from .encoder.model import AudioEncoder
from huggingface_hub import snapshot_download
class TTSCodec:

    def __init__(self):

        d_path = snapshot_download("YatharthS/MiraTTS")
        d_path = f"{d_path}/decoders"
        self.audio_decoder = AudioDecoder(d_path)
        self.audio_encoder = AudioEncoder(d_path)


    def encode(self, audio, encode_semantic=False, duration=8):
        if encode_semantic:
            speech_tokens, context_tokens = self.audio_encoder.encode(audio, True, duration=duration)
            return speech_tokens, context_tokens
        else:
            context_tokens = self.audio_encoder.encode(audio, False, duration=duration)
            return context_tokens
            
    def process_audio(self, wav, wav2):
        wav = wav.cpu().numpy()

        weight_1, weight_2 = self.weight_1, self.weight_2
        mixed_audio = (wav * weight_1) + (wav2 * weight_2)
        return mixed_audio
    
    def format_prompt(self, text, context_tokens, extra_tokens, semantic_tokens=None, transcript=None):
        if semantic_tokens:
            prompt = f"<|task_tts|><|start_text|>{text}<|end_text|><|context_audio_start|>{context_tokens}<|context_audio_end|><|prompt_speech_start|>{semantic_tokens}"
        else:
            prompt = f"<|task_tts|><|start_text|>{text}<|end_text|><|context_audio_start|>{context_tokens}<|context_audio_end|><|prompt_speech_start|>"
        return prompt

    def c_cache(self):
        gc.collect()
        torch.cuda.empty_cache()
        
    def decode(self, speech_tokens, context_tokens, test_var=None):

        wav = self.audio_decoder.detokenize(
            context_tokens,
            speech_tokens,
        )

        return wav
