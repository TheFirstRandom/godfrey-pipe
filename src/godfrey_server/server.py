import asyncio
from typing import Literal, cast

from rich.console import Console
from wyoming.server import AsyncEventHandler, AsyncServer
from wyoming.audio import AudioChunk
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

            if self.models["Silero VAD"].predict(raw) is False:
                self.change_state("processing")
                self.models["Silero VAD"].reset()

                self._pipeline_task = asyncio.create_task(self.run_pipeline())
                cast(asyncio.Task, self._pipeline_task).add_done_callback(self._on_pipeline_done)

        elif self.state == "processing":
            return

        else:
            for i in range(2):
                frame = raw[i:i+2]
                prediction = self.models["openWakeWord"].predict(frame)
                if prediction is True:
                    self.change_state("listening")
                    self.models["openWakeWord"].reset()

    async def run_pipeline(self):
        self.console.print("[1/3] Transcribing audio...")
        user_text = self.models["faster-whisper"].transcribe(self.recording)
        self.console.print("Result:", user_text[:100] if len(user_text) > 100 else user_text)

        self.console.print("[2/3] Generating answer...")
        answer = self.models["Qwen 3.6"].answer(user_text)
        self.console.print("Result:", answer[:100] if len(answer) > 100 else answer)

        self.console.print("[3/3] Generating voice...")
        audio = self.models["Kokoro TTS"].transcribe()
        self.console.print("Result:", audio[0] if audio else "no output")

        if audio:
            await self.send_answer(audio)

    async def send_answer(self, audio):
        # Sende hier das audio

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


async def start_server():
    server = AsyncServer.from_uri("tcp://0.0.0.0:10700")
    await server.run(VoiceHandler)