import asyncio
import gc
import os
from functools import partial
from typing import Literal, cast

import numpy as np
import torch
from pydub import AudioSegment
from rich.console import Console
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.server import AsyncEventHandler, AsyncServer

from godfrey_server.models import *


class VoiceHandler(AsyncEventHandler):
    def __init__(self, models: dict, console: Console, *args, **kwargs):
        """The wyoming connection to a client and provides the pipeline for processing audio and answering requests.

        Once executed with async, this class will continuously wait for audio chunks, check them for the wakeword
        and run the pipeline, if the wakeword is recognized. The pipeline will generate a answer in voice and
        send it back to the client.

        Attributes:
            models (dict): A dict with all loaded models.
            console (Console): A rich console used for formatted text output.
            openwakeword_threshold (float): Threshold for detecting wakewords. Derived from OPENWAKEWORD_THRESHOLD.
                Defaults to 0.2.
            state (Literal["waiting", "listening", "processing"]): The state which the server is currently in.
                - waiting: The server is waiting for the recognition of the wakeword.
                - listening: The server recognized the wakeword and is now listening for your input.
                    The VAD will detect your speech's end.
                - processing: The pipeline is processing your request. No new requests will be processed during this.

        Args:
            models: A dict containing all model instances loaded by the main module.
            console: A rich console for formatted text output.
            *args: Arguments passed to the AsyncEventHandler parent.
            **kwargs: Keyword Arguments passed to the AsyncEventHandler parent.
        """
        self.models = models
        self.console = console
        super().__init__(*args, **kwargs)

        self.state: Literal["waiting", "listening", "processing"] = "waiting"
        self.recording: list[bytes] = []
        self._pipeline_task: asyncio.Task | None = None
        self.openwakeword_threshold = os.getenv("OPENWAKEWORD_THRESHOLD", 0.2)
        self.play_intro = True

    def _change_state(self, state: Literal["waiting", "listening", "processing"]):
        """Change the servers state to the requested value and print the new value.

        Args:
            state(Literal["waiting", "listening", "processing"]): The new state for the server to enter.
        """
        self.state = state
        self.console.print(f"Server changed state to {state}")

    async def handle_event(self, event: Event) -> bool:
        """This method gets executed by ``AsyncEventHandler`` on receiving a wyoming event.

        It starts ``handle_audio_chunk`` if the event was audio. If this is the first time the client sends an event,
        an intro is played through ``self.play_notification``.

        Args:
            event (Event): The event received by the ``AsyncEventHandler``.
        """
        if self.play_intro:
            self.console.print("Playing intro")
            await self.play_notification("godfrey_intro.wav", volume=0.8)
            self.play_intro = False

        if AudioChunk.is_type(event.type):
            chunk = AudioChunk.from_event(event)
            await self.handle_audio_chunk(chunk.audio)

        return True

    async def handle_audio_chunk(self, raw: bytes):
        """Processes a received audio chunk. The action depends on the current server state.

        The different actions for all three states are:
            - listening: The new audio chunk gets appended to ``self.recording``.
                SileroVAD predicts an end of speech in the chunk. If the prediction is ``True``,
                it plays a notification sound and changes the server state to processing.
            - processing: The function just returns.
            - waiting: openWakeWord predicts the occurrence of the wakeword with the new chunk.
                If it predicts the wakeword, it will play a notification sound, change the server state to listening,
                reset the SileroVAD model and start ``run_pipeline()`` as async task.

        Args:
            raw (bytes): The received audio chunk.
        """
        if self.state == "listening":
            self.recording.append(raw)

            samples = data.pcm16_bytes_to_float32(raw)
            if self.models["Silero VAD"].predict(samples) is False:
                await self.play_notification("Notification2.wav", volume=0.6)
                self._change_state("processing")
                self.models["Silero VAD"].reset()

                self._pipeline_task = asyncio.create_task(self.run_pipeline())
                cast(asyncio.Task, self._pipeline_task).add_done_callback(self._on_pipeline_done)

        elif self.state == "processing":
            return

        else:
            samples = data.pcm16_bytes_to_int16_array(raw)
            prediction = self.models["openWakeWord"].predict(samples)
            if prediction > float(self.openwakeword_threshold):
                await self.play_notification("Notification1.wav", volume=0.6)
                self.console.print(f"Predicted the wakeword with [cyan]{prediction}[/cyan]")
                self._change_state("listening")
                self.models["openWakeWord"].reset()

    async def run_pipeline(self):
        """Runs the Godfrey pipeline, which processes the audio input of the user and sends the answer.

        First, faster-whisper transcribes the users audio to text. After that, Qwen is generating an answer and
        running tools, which were detected in the input. Kokoro is then generating the text as audio.
        At the end it sends the audio with ``send_answer`` to the client.
        """
        self.console.print("[1/3] Transcribing audio...")
        audio_array = data.join_chunks_to_float32(self.recording)
        user_text = self.models["faster-whisper"].transcribe(audio_array)
        self.console.print(f"Result: [italic]{user_text}[/italic]")

        self.console.print("[2/3] Generating answer...")
        answer = self.models["Qwen 3"].answer(user_text)
        self.console.print(f"Result: [italic]{answer}[/italic]")

        self.console.print("[3/3] Generating voice...")
        # Fix: transcribe() requires the text to synthesize; it was being
        # called with no arguments at all, which would raise a TypeError.
        audio = self.models["Kokoro TTS"].transcribe(answer)

        if audio:
            await self.send_answer(audio)

    async def send_answer(self, audio):
        """
        Sends an answer audio from Kokoro to the client over the Wyoming protocol.

        Args:
            audio: The audio returned by ``models.KokoroTTS.transcribe()``.
        """
        # Fix: Implemented actual audio streaming back to the client over the Wyoming protocol.
        sample_rate = 16000
        sample_width = 2  # 16-bit PCM
        channels = 1

        await self.write_event(AudioStart(rate=sample_rate, width=sample_width, channels=channels).event())

        for result in audio:
            samples = np.asarray(result.audio, dtype=np.float32)
            samples_16k = data.resample_by_ratio(samples, 2, 3)
            pcm_bytes = data.float32_to_pcm16_bytes(samples_16k)

            await self.write_event(
                AudioChunk(rate=sample_rate, width=sample_width, channels=channels, audio=pcm_bytes).event()
            )

        await self.write_event(AudioStop().event())

    def _on_pipeline_done(self, task: asyncio.Task):
        """Cleans up after the pipeline was run.

        Has to be registered with ``pipeline_task.add_done_callback(self._on_pipeline_done)`` on the pipeline task.
        Is sets the server state back to waiting, resets the recording and the pipeline task.
        As well, it frees up unused cache VRAM memory to prevent a memory leak.

        Args:
            task (Task): The async task. Gets provided by ``add_done_callback``.
        """
        task.result()
        self._change_state("waiting")
        self.recording = []
        self._pipeline_task = None

        # Free unused memory
        gc.collect()
        torch.cuda.empty_cache()

    async def play_notification(self, filename: str, volume: float = 1.0):
        """Plays a notification on the client. This can be a sound or even longer sequences of audio.

        It looks for the notification sound file in ``NOTIFICATION_SOUNDS_PATH``.

        Args:
            filename (str): The filename of the sound to play.
            volume (float): The volume for the sound from 0.0 to 1.0.
        """
        # [KI] Spielt eine WAV oder MP3 Datei über den Wyoming-Client ab.

        path = data.path_from_env_var("NOTIFICATION_SOUNDS_PATH") / filename
        audio = AudioSegment.from_file(path)

        audio = audio.set_channels(1).set_sample_width(2)

        pcm_bytes = data.pcm16_bytes_resample_to_rate(
            audio.raw_data,
            audio.frame_rate,
            volume=volume,
            target_rate=16000
        )

        await self.write_event(AudioStart(rate=16000, width=2, channels=1).event())

        chunk_size = 512 * 2
        for i in range(0, len(pcm_bytes), chunk_size):
            await self.write_event(
                AudioChunk(rate=16000, width=2, channels=1, audio=pcm_bytes[i:i + chunk_size]).event()
            )

        await self.write_event(AudioStop().event())


async def start_server(models: dict, console: Console):
    """Starts the Wyoming server with the ``VoiceHandler``."""
    server = AsyncServer.from_uri("tcp://0.0.0.0:10700")
    await server.run(partial(VoiceHandler, models, console))
