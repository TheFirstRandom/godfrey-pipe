from rich.console import Console

from godfrey_server import __version__





def cli():
    console = Console(highlight=False)
    console.print("Waking up Godfrey...", style="bold")
    console.print(f"Server version: {__version__}")

    console.print("Loading [cyan]faster-whisper[/cyan]...")

    # Start & load faster-whisper

    console.print("Loading [cyan]qwen3.6[/cyan]...")

    # Start & load qwen

    console.print("Loading [cyan]kokoro[/cyan]...")

    # Start & load kokoro
