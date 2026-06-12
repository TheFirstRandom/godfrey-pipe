import os
from pathlib import Path

import openwakeword
import silero_vad
import faster_whisper
import ollama
import kokoro


class OpenWakeWord:
    def __init__(self):
        model_path_var = os.environ.get("OPENWAKEWORD_MODEL_PATH")
        if not model_path_var:
            raise ValueError("Missing value for OPENWAKEWORD_MODEL_PATH")

        self.model_path = Path(model_path_var)
        if not self.model_path.exists():
            raise FileExistsError(f"File not found: {self.model_path}")

        self.model = openwakeword.model.Model(wakeword_models=[str(self.model_path)])

    def predict(self, frame) -> bool:
        prediction = self.model.predict(frame)
        return next(iter(prediction.values()))


class SileroVAD:
    def __init__(self):
        self.model = silero_vad.load_silero_vad()
        self.iterator = silero_vad.VADIterator(self.model, sampling_rate=16000)


class FasterWhisper:
    def __init__(self):
        self.model = faster_whisper.WhisperModel("large-v3", device="cpu", compute_type="int8")

    def transcribe(self, audio):
        segments, info = self.model.transcribe(audio)

        return " ".join(segment.text for segment in segments)


class Qwen:
    def __init__(self):
        self.chat = ollama.chat(
            model="qwen3.6",
            messages=[],
            stream=True,
            keep_alive=-1
        )


class KokoroTTS:
    def __init__(self):
        self.pipeline = kokoro.KPipeline(lang_code="a")
