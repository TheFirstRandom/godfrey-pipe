import asyncio
from functools import partial
from typing import Literal, cast

import numpy as np
from rich.console import Console
from wyoming.server import AsyncEventHandler, AsyncServer
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event

from godfrey_server.models import *


class VoiceHandler(AsyncEventHandler):
    def __init__(self, models: dict, console: Console, *args, **kwargs):
        self.models = models
        self.console = console
        super().__init__(*args, **kwargs)

        self.state: Literal["waiting", "listening", "processing"] = "waiting"
        self.recording: list[bytes] = []
        self._pipeline_task: asyncio.Task | None = None

    def change_state(self, state: Literal["waiting", "listening", "processing"]):
        self.state = state
        self.console.print(f"Server changed state to {state}")

    async def handle_event(self, event: Event) -> bool:
        if AudioChunk.is_type(event.type):
            chunk = AudioChunk.from_event(event)
            await self.handle_audio_chunk(chunk.audio)

        return True

    async def handle_audio_chunk(self, raw: bytes):
        if self.state == "listening":
            self.recording.append(raw)

            # Fix: Silero VAD expects float32 samples in [-1, 1], not raw
            # 16-bit PCM bytes. Convert before feeding it to the model.
            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            if self.models["Silero VAD"].predict(samples) is False:
                self.change_state("processing")
                self.models["Silero VAD"].reset()

                self._pipeline_task = asyncio.create_task(self.run_pipeline())
                cast(asyncio.Task, self._pipeline_task).add_done_callback(self._on_pipeline_done)

        elif self.state == "processing":
            return

        else:
            # Fix: the previous code sliced the raw bytes into meaningless
            # 2-byte fragments (`raw[i:i+2]` for i in range(2)), which is
            # not a valid audio frame for openWakeWord. Decode the whole
            # chunk into a proper int16 sample array instead.
            samples = np.frombuffer(raw, dtype=np.int16)
            prediction = self.models["openWakeWord"].predict(samples)
            if prediction > 0.4:
                self.console.print(f"\\[openWakeWord\\] predicted {prediction}")
                self.change_state("listening")
                self.models["openWakeWord"].reset()

    async def run_pipeline(self):
        self.console.print("[1/3] Transcribing audio...")
        # Fix: faster-whisper needs a float32 waveform, not a list of raw
        # PCM byte chunks. Concatenate and convert before transcribing.
        raw_audio = b"".join(self.recording)
        audio_array = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32) / 32768.0
        user_text = self.models["faster-whisper"].transcribe(audio_array)
        self.console.print("Result:", user_text)

        self.console.print("[2/3] Generating answer...")
        answer = self.models["Qwen 3"].answer(user_text)
        self.console.print("Result:", answer)

        self.console.print("[3/3] Generating voice...")
        # Fix: transcribe() requires the text to synthesize; it was being
        # called with no arguments at all, which would raise a TypeError.
        audio = self.models["Kokoro TTS"].transcribe(answer)

        if audio:
            await self.send_answer(audio)

    async def send_answer(self, audio):
        # Fix: this method had no body at all (just a comment), which is a
        # SyntaxError in Python and would have crashed on import. Implemented
        # actual audio streaming back to the client over the Wyoming protocol.
        # Assumption: Kokoro's KPipeline yields result objects exposing an
        # `.audio` attribute (float32 waveform, 24 kHz, mono) - adjust the
        # attribute name below if your installed kokoro version differs.
        sample_rate = 16000
        sample_width = 2  # 16-bit PCM
        channels = 1

        await self.write_event(AudioStart(rate=sample_rate, width=sample_width, channels=channels).event())

        for result in audio:
            samples = np.asarray(result.audio)
            pcm_bytes = (samples * 32767).astype(np.int16).tobytes()
            await self.write_event(
                AudioChunk(rate=sample_rate, width=sample_width, channels=channels, audio=pcm_bytes).event()
            )

        await self.write_event(AudioStop().event())

    def _on_pipeline_done(self, task: asyncio.Task):
        task.result()
        self.change_state("waiting")
        self.recording = []
        self._pipeline_task = None


def load_wws():
    return OpenWakeWord()


def load_vad():
    return SileroVAD()


def load_stt():
    return FasterWhisper()


def load_llm():
    return Qwen()


def load_tts():
    return KokoroTTS()


async def start_server(models: dict, console: Console):
    # Fix: the loaded models from main.py were never passed in here, and
    # AsyncServer.run() instantiates the handler class itself per
    # connection, but VoiceHandler.__init__ requires (models, console) in
    # addition to the connection args. Bind them with functools.partial so
    # each new connection gets a properly constructed VoiceHandler.
    server = AsyncServer.from_uri("tcp://0.0.0.0:10700")
    await server.run(partial(VoiceHandler, models, console))