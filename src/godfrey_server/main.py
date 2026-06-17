from concurrent.futures import ThreadPoolExecutor, as_completed
from importlib.metadata import version

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from godfrey_server import server


def cli():
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
            ("openWakeWord", server.load_wws),
            ("Silero VAD", server.load_vad),
            ("faster-whisper", server.load_stt),
            ("Qwen 3.6", server.load_llm),
            ("Kokoro TTS", server.load_tts),
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


if __name__ == "__main__":
    cli()