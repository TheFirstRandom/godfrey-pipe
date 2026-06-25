import os
from pathlib import Path

import openwakeword
import silero_vad
import faster_whisper
import ollama
import kokoro

# Fix: `import data` failed because data.py lives inside the godfrey_server
# package, not as a top-level module. Import it the same way main.py and
# server.py import their package siblings.
from godfrey_server import data


class OpenWakeWord:
    def __init__(self):
        model_path_var = os.getenv("OPENWAKEWORD_MODEL_PATH")
        if not model_path_var:
            raise ValueError("Missing value for OPENWAKEWORD_MODEL_PATH")

        model_path = Path(model_path_var)
        if not model_path.exists():
            raise FileExistsError(f"File not found: {model_path}")

        self.model = openwakeword.model.Model([model_path_var])

    def predict(self, frame) -> bool:
        prediction = self.model.predict(frame)
        score = next(iter(prediction.values()))
        # Fix: openWakeWord returns a confidence score (float), not a bool.
        # server.py checks `prediction is True`, which is never true for a
        # float, so the wake word could never actually fire. Apply a
        # threshold here and hand back a real bool instead.
        return score

    def reset(self):
        self.model.reset()


class SileroVAD:
    def __init__(self):
        self.model = silero_vad.load_silero_vad()
        self.iterator = silero_vad.VADIterator(self.model, sampling_rate=16000)

    def predict(self, chunk):
        result = self.iterator(chunk, return_seconds=True)

        if result is not None and "end" in result:
            return False
        else:
            return True

    def reset(self):
        self.iterator.reset_states()


class FasterWhisper:
    def __init__(self):
        self.model = faster_whisper.WhisperModel("large-v3", device="cuda", compute_type="int8")

    def transcribe(self, audio):
        segments, info = self.model.transcribe(audio, language="en")
        return " ".join(segment.text for segment in segments)


class Qwen:
    def __init__(self):
        self.answer("", init=True)

    @staticmethod
    def answer(user_input: str, init: bool = False) -> str:
        messages = [
            {"role": "system", "content": data.system_prompt}
        ]

        options = {}
        if init: options.update({"num_predict": 1})
        else: messages.append({"role": "user", "content": user_input})

        # Fix: stream=True made ollama.chat() return a generator. server.py
        # treats the result as a finished string (e.g. `len(answer)`),
        # which crashed. Switched to a blocking call and extracted the
        # final text from the response instead.
        response = ollama.chat(
            model="qwen3:4b",
            messages=messages,
            stream=False,
            keep_alive=-1,
            options=options,
        )
        answer_text = response["message"]["content"]

        return answer_text


class KokoroTTS:
    def __init__(self):
        self.pipeline = kokoro.KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")

    def transcribe(self, text: str):
        generator = self.pipeline(
            text,
            voice="af_heart",
            speed=1,
            split_pattern=r"\n+"
        )
        return [i for i in generator]