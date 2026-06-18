# import os
# from pathlib import Path

import openwakeword
import silero_vad
import faster_whisper
import ollama
import kokoro

import data


class OpenWakeWord:
    def __init__(self):
        # model_path_var = os.environ.get("OPENWAKEWORD_MODEL_PATH")
        # if not model_path_var:
        #     raise ValueError("Missing value for OPENWAKEWORD_MODEL_PATH")
        #
        # self.model_path = Path(model_path_var)
        # if not self.model_path.exists():
        #     raise FileExistsError(f"File not found: {self.model_path}")

        openwakeword.utils.download_models()

        self.model = openwakeword.model.Model()

    def predict(self, frame) -> bool:
        prediction = self.model.predict(frame)
        return next(iter(prediction.values()))

    def reset(self):
        self.model.reset()


class SileroVAD:
    def __init__(self):
        self.model = silero_vad.load_silero_vad()
        self.iterator = silero_vad.VADIterator(self.model, sampling_rate=16000)

    def predict(self, chunk):
        result = self.iterator(chunk, return_seconds=True)
        return False if "end" in result else True

    def reset(self):
        self.iterator.reset_states()


class FasterWhisper:
    def __init__(self):
        self.model = faster_whisper.WhisperModel("large-v3", device="cpu", compute_type="int8")

    def transcribe(self, audio):
        segments, info = self.model.transcribe(audio, language="en")
        return " ".join(segment.text for segment in segments)


class Qwen:
    def __init__(self):
        self.messages = [
            {"role": "system", "content": data.system_prompt}
        ]
        self.answer("", init=True)

    def answer(self, user_input: str, init: bool = False):
        messages = self.messages
        if not init:
            messages.append({"role": "user", "content": user_input})

        return ollama.chat(
            model="qwen3.6",
            messages=messages,
            stream=True,
            keep_alive=-1
        )


class KokoroTTS:
    def __init__(self):
        self.pipeline = kokoro.KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")

    def transcribe(self, text: str):
        generator = self.pipeline(
            text,
            voice="af_heart",
            speed=0.75,
            split_pattern=r"\n+"
        )
        return [i for i in generator]