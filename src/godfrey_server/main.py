import threading
from typing import cast

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, TaskID

from godfrey_server import __version__
from godfrey_server import server


def cli():
    console = Console(highlight=False)
    console.print("Waking up Godfrey...", style="bold")
    console.print(f"Server version: {__version__}\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        threads = {
            "openWakeWord": {
                "track": progress.add_task("Loading [cyan]openWakeWord[/cyan]...", total=None),
                "thread": threading.Thread(target=server.load_wws),
            },
            "Silero VAD": {
                "track": progress.add_task("Loading [cyan]Silero VAD[/cyan]...", total=None),
                "thread": threading.Thread(target=server.load_vad),
            },
            "faster-whisper": {
                "track": progress.add_task("Loading [cyan]faster-whisper[/cyan]...", total=None),
                "thread": threading.Thread(target=server.load_stt),
            },
            "Qwen 3.6": {
                "track": progress.add_task("Loading [cyan]Qwen 3.6[/cyan]...", total=None),
                "thread": threading.Thread(target=server.load_llm),
            },
            "Kokoro TTS": {
                "track": progress.add_task("Loading [cyan]Kokoro TTS[/cyan]...", total=None),
                "thread": threading.Thread(target=server.load_tts),
            },
        }

        for name, data in threads.items():
            task_id = cast(TaskID, data["track"])
            thread = cast(threading.Thread, data["thread"])

            thread.start()
            progress.update(task_id)

        for name, data in threads.items():
            task_id = cast(TaskID, data["track"])
            thread = cast(threading.Thread, data["thread"])

            thread.join()


if __name__ == "__main__":
    cli()
