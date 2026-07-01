import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from importlib.metadata import version

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from godfrey_server import server, models


def cli():
    """The entrypoint for the Godfrey server. Loads all models and starts the server.

    It also shows the progress while loading models. The models load parallel by using a ``ThreadPoolExecutor``.
    """
    console = Console(highlight=False)
    console.print("Waking up Godfrey...", style="bold")
    console.print(f"Server version: {version('godfrey-pipe')}\n")

    # everything inside this Progress written with help of AI (see index 3 in docs)
    with Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            TimeElapsedColumn(),
            console=console
    ) as progress:
        tasks = [
            ("openWakeWord", models.OpenWakeWord),
            ("Silero VAD", models.SileroVAD),
            ("faster-whisper", models.FasterWhisper),
            ("Qwen 3", models.Qwen),
            ("Kokoro TTS", models.KokoroTTS),
        ]

        with ThreadPoolExecutor() as executor:
            # Future -> (label, task_id) mapping
            future_map = {
                executor.submit(fn): (
                    label,
                    progress.add_task(f"Loading {label}...", total=1)
                ) for label, fn in tasks
            }

            results = {}
            for future in as_completed(future_map):
                label, task_id = future_map[future]
                results[label] = future.result()

                progress.update(
                    task_id,
                    completed=1,
                    description=f"[green]✓[/green] {label} loaded"
                )

    console.print("[green]✓[/green] Server running", style="bold")
    asyncio.run(server.start_server(results, console))


if __name__ == "__main__":
    cli()