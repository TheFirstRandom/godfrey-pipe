import asyncio

from wyoming.server import AsyncServer, AsyncEventHandler
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event

from godfrey_server.models import *


# Claude
class AudioReceiver(AsyncEventHandler):
    async def handle_event(self, event: Event):
        if AudioStart.is_type(event.type):
            audio = AudioStart.from_event(event)
            return {
                "type": "start",
                "sample_rate": audio.rate,
            }

        elif AudioChunk.is_type(event.type):
            chunk = AudioChunk.from_event(event)
            return {
                "type": "chunk",
                "audio_chunk": chunk.audio
            }

        elif AudioStop.is_type(event.type):
            return {"type": "stop"}

        else:
            raise ValueError("Unknown wyoming event type")


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