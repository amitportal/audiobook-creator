import numpy as np
from pathlib import Path
from typing import Optional, List
from abc import ABC, abstractmethod

# Mocking the base class logic here to test it without dependencies
class TTSEngine(ABC):
    def __init__(self):
        self.max_text_length = 500
        self.sample_rate = 1000
        self.model = None

    @abstractmethod
    def _synthesize_single(self, text: str) -> np.ndarray:
        pass

    def synthesize(self, text: str) -> np.ndarray:
        if '\n' in text or len(text) > self.max_text_length:
            return self._split_and_synthesize(text)
        return self._synthesize_single(text)

    def _split_and_synthesize(self, text: str) -> np.ndarray:
        import re
        if '\n' in text:
            lines = text.split('\n')
            audio_segments = []
            for line in lines:
                line = line.strip()
                if not line: continue
                audio_segments.append(self.synthesize(line))
                audio_segments.append(np.zeros(10))
            return np.concatenate(audio_segments[:-1]) if audio_segments else np.zeros(0)

        sentences = re.split(r'(?<=[.!?])\s+', text)
        if len(sentences) == 1 and len(sentences[0]) > self.max_text_length:
            sentences = re.split(r'(?<=[,;])\s+|\s+', text)
            
        audio_segments = []
        current_chunk = ""
        for part in sentences:
            if len(part) > self.max_text_length:
                if current_chunk:
                    audio_segments.append(self._synthesize_single(current_chunk))
                    audio_segments.append(np.zeros(5))
                    current_chunk = ""
                for i in range(0, len(part), self.max_text_length):
                    sub_part = part[i:i + self.max_text_length]
                    audio_segments.append(self._synthesize_single(sub_part))
                    if i + self.max_text_length < len(part):
                        audio_segments.append(np.zeros(5))
                continue
            if len(current_chunk) + len(part) + 1 < self.max_text_length:
                current_chunk += " " + part if current_chunk else part
            else:
                if current_chunk:
                    audio_segments.append(self._synthesize_single(current_chunk))
                    audio_segments.append(np.zeros(5))
                current_chunk = part
        if current_chunk:
            audio_segments.append(self._synthesize_single(current_chunk))
        return np.concatenate(audio_segments) if audio_segments else np.zeros(0)

class MockEngine(TTSEngine):
    def __init__(self, max_len=10):
        super().__init__()
        self.max_text_length = max_len
        self.sample_rate = 1000
        self.calls = []

    def load_model(self):
        self.model = True

    def _synthesize_single(self, text: str) -> np.ndarray:
        self.calls.append(text)
        return np.zeros(100)

def test_splitting():
    # Test 1: Simple split
    engine = MockEngine(max_len=10)
    engine.synthesize("Hello world this is long")
    print(f"Calls (max 10): {engine.calls}")
    assert all(len(c) <= 10 for c in engine.calls)
    
    # Test 2: Line breaks
    engine = MockEngine(max_len=100)
    engine.calls = []
    engine.synthesize("Line 1\nLine 2")
    print(f"Calls (with newline): {engine.calls}")
    assert "Line 1" in engine.calls
    assert "Line 2" in engine.calls

    # Test 3: Very long word
    engine = MockEngine(max_len=5)
    engine.calls = []
    engine.synthesize("Supercalifragilistic")
    print(f"Calls (long word, max 5): {engine.calls}")
    assert all(len(c) <= 5 for c in engine.calls)

if __name__ == "__main__":
    try:
        test_splitting()
        print("\nVerification successful! Splitting logic is robust.")
    except Exception as e:
        print(f"\nVerification failed: {e}")
