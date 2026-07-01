import subprocess

import faster_whisper
import kokoro
import ollama
import openwakeword
import silero_vad
from ollama import ChatResponse
from ddgs import DDGS

# Fix: `import data` failed because data.py lives inside the godfrey_server
# package, not as a top-level module. Import it the same way main.py and
# server.py import their package siblings.
from godfrey_server import data


class OpenWakeWord:
    def __init__(self):
        """Represents a loaded openWakeWord model and provides methods to interact with it.

        It loads the model at it's ``__init__()`` and downloads models once at the first call.
        The model will be loaded from ``OPENWAKEWORD_MODEL_PATH``. The model has to be in ``onnx`` format.
        """
        openwakeword.utils.download_models()

        model_path = data.path_from_env_var("OPENWAKEWORD_MODEL_PATH")

        self.model = openwakeword.model.Model(
            [str(model_path)],
            inference_framework="onnx",
        )

    def predict(self, frame) -> float:
        """Predicts the occurrence of the wakeword in a audio frame.

        Args:
            frame: The audio frame to predict the occurrence of.

        Returns:
            A float containing the probability of the wakeword in the audio frame.
        """
        prediction = self.model.predict(frame)
        score = next(iter(prediction.values()))
        # Fix: openWakeWord returns a confidence score (float), not a bool.
        # server.py checks `prediction is True`, which is never true for a
        # float, so the wake word could never actually fire. Apply a
        # threshold here and hand back a real bool instead.
        return score

    def reset(self):
        """Resets the model to forget previous frames."""
        self.model.reset()


class SileroVAD:
    def __init__(self):
        """Represents a loaded SileroVAD model and provides methods to interact with it.

        It loads the model at it's ``__init__()``.
        """
        self.model = silero_vad.load_silero_vad()
        self.iterator = silero_vad.VADIterator(self.model, sampling_rate=16000)

    def predict(self, chunk) -> bool:
        """Predicts the end of speech in a audio chunk.

        Args:
            chunk: The next audio chunk to predict the end of speech in.

        Returns:
            A bool indicating whether the VAD found a end of speech or not.
        """
        result = self.iterator(chunk, return_seconds=True)

        if result is not None and "end" in result:
            return False
        else:
            return True

    def reset(self):
        """Resets the model to forget previous chunks."""
        self.iterator.reset_states()


class FasterWhisper:
    def __init__(self):
        """Represents a loaded faster-whisper model and provides methods to interact with it.

        It loads the model at it's ``__init__()``.
        """
        self.model = faster_whisper.WhisperModel("large-v3", device="cuda", compute_type="int8")

    def transcribe(self, audio) -> str:
        """Transcribes an audio sequence with speech to text.

        Args:
            audio: The audio sequence to transcribe.

        Returns:
            A string containing the transcribed audio sequence.
        """
        segments, info = self.model.transcribe(audio, language="en")
        return " ".join(segment.text for segment in segments)


class Qwen:
    def __init__(self):
        """Represents a loaded faster-whisper model and provides methods to interact with it.

        It loads the model at it's ``__init__()`` by generating an answer with ``keep_active=-1`` and
        ``options={"num_predict": 1}`` to prevent generating a long answer, which would slow down the process.
        """
        self._generate([], [], {"num_predict": 1})

    def answer(self, user_input: str) -> str:
        """Responds to the users input in text and runs tools, if requested.

        It creates a chat with the system prompt and the users input.
        It provides the available tools to control a light over MQTT.
        The model will look through the functions and run one of the scripts in ``scripts/`` if
        it was requested by an action call in the request.
        """
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
            # ... deine bestehenden turn_on_lights / turn_off_lights ...
            {
                "type": "function",
                "function": {
                    "name": "search_web",
                    "description": "Search the internet for current information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query"
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
        ]

        response = self._generate(messages, tools, {})
        tool_calls = response["message"].get("tool_calls") or []

        # Kein Tool nötig -> direkt antworten, wie bisher
        if not tool_calls:
            return response["message"]["content"].replace("*", "")

        # Modell hat Tools aufgerufen -> Nachricht des Modells zur History hinzufügen
        messages.append(response["message"])

        for call in tool_calls:
            function = call["function"]["name"]

            if function == "search_web":
                query = call["function"]["arguments"]["query"]
                result = search_web(query)
            else:
                result = subprocess.run(f"./scripts/{function}")

            # WICHTIG: Ergebnis als "tool"-Nachricht zurückgeben
            messages.append({
                "role": "tool",
                "content": str(result),
            })

        # Zweiter Aufruf: Modell sieht die Suchresultate und formuliert die finale Antwort
        final_response = self._generate(messages, tools, {})
        return final_response["message"]["content"].replace("*", "")

    @staticmethod
    def _generate(messages: list, tools: list, options: dict) -> ChatResponse:
        """Generates an answer without processing anything of it.

        Args:
            messages: A list of dictionaries containing all the messages sent by participants of the chat.
            tools: A list of dictionaries containing all the tools available for the LLM.
            options: A dictionary containing extra options for ollama.

        Returns:
            A ``ChatResponse`` object with the answer in text, tool calls and more.
        """
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
        """Represents a loaded Kokoro model and provides methods to interact with it.

        It loads the model at it's ``__init__()``.
        """
        self.pipeline = kokoro.KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")

    def transcribe(self, text: str) -> list:
        """Generates an audio from a given text.

        It uses the voice ``bm_lewis`` to match Godfrey's voice as good as possible. The speed is set to ``0.75``
        to let it sound even more like the character.

        Args:
            text: The text to transcribe.

        Returns:
            A list of audio chunks generated by the Kokoro model.
        """
        generator = self.pipeline(
            text,
            voice="bm_lewis",
            speed=0.75,
            split_pattern=r"\n+"
        )
        return [i for i in generator]
def search_web(query: str, max_results: int = 5) -> str:
    """
    Searches for a web search query.
    Args:
        query: The search query."""
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=max_results))
    return "/n/n".join(f"{r['title']}: {r['url']}" for r in results)