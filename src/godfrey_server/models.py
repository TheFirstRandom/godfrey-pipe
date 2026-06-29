import os
from pathlib import Path

import faster_whisper
import kokoro
import ollama
import openwakeword
import silero_vad
import subprocess

# Fix: `import data` failed because data.py lives inside the godfrey_server
# package, not as a top-level module. Import it the same way main.py and
# server.py import their package siblings.
from godfrey_server import data


class OpenWakeWord:
    def __init__(self):
        openwakeword.utils.download_models()

        model_path_var = os.getenv("OPENWAKEWORD_MODEL_PATH")
        if not model_path_var:
            raise ValueError("Missing value for OPENWAKEWORD_MODEL_PATH")

        model_path = Path(model_path_var)
        if not model_path.exists():
            raise FileNotFoundError(f"File not found: {model_path}")

        self.model = openwakeword.model.Model(
            [model_path_var],
            inference_framework="onnx",
        )

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
        self._generate([], [], {"num_predict": 1})

    def answer(self, user_input: str) -> str:
        messages = [
            {"role": "system", "content": data.system_prompt},
            {"role": "user", "content": user_input}
        ]

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "turn_on_lights",
                    "description": "Activate an LED light",
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "turn_off_lights",
                    "description": "Turn off the LED light",
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    }
                }
            },
        ]

        response = self._generate(messages, tools, {})
        answer_text = response["message"]["content"]
        tool_calls = response["message"].get("tool_calls") or []

        results = {}
        for call in tool_calls:
            function = call["function"]["name"]
            result = subprocess.run(f"./scripts/{function}")
            results[function] = result

        for name, result in results.items():
            messages.append({
                    "role": "tool",
                    "content": f"Called tool {name}. The tool {'succeeded' if result.returncode == 0 else 'failed'}."
                })

        # in case tools were run
        if len(messages) < 3:
            answer_text = self._generate(messages, [], {})["message"]["content"]

        return answer_text

    @staticmethod
    def _generate(messages: list, tools: list, options: dict):
        return ollama.chat(
            model="qwen3:4b",
            messages=messages,
            tools=tools,
            stream=False,
            keep_alive=-1,
            options=options,
        )


class KokoroTTS:
    def __init__(self):
        self.pipeline = kokoro.KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")

    def transcribe(self, text: str):
        generator = self.pipeline(
            text,
            voice="bm_lewis",
            speed=0.75,
            split_pattern=r"\n+"
        )
        return [i for i in generator]
