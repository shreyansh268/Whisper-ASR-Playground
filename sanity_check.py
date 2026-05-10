"""
Sanity Check — distil-whisper/librispeech_long
===============================================
Streams the audio sample from the validation split and transcribes
it with whisper-tiny. No reference text — transcription only.

Usage:
    python sanity_check.py

Install additional deps:
    pip install datasets soundfile
"""

import sys
import io
from pathlib import Path

import numpy as np

try:
    from datasets import load_dataset, Audio
except ImportError:
    print("ERROR: datasets not installed. Run: pip install datasets")
    sys.exit(1)

try:
    import soundfile as sf
except ImportError:
    print("ERROR: soundfile not installed. Run: pip install soundfile")
    sys.exit(1)

from asr_tester import load_model, transcribe, print_result, save_result, RICH

if RICH:
    from rich.console import Console
    from rich.panel import Panel
    console = Console()


DATASET_ID = "distil-whisper/librispeech_long"
SUBSET     = "clean"
SPLIT      = "validation"


def run():
    if RICH:
        console.print(Panel.fit(
            f"[bold cyan]Sanity Check — whisper-tiny[/bold cyan]\n"
            f"[dim]Dataset : {DATASET_ID}  |  {SUBSET}/{SPLIT}[/dim]",
            border_style="cyan"
        ))
    else:
        print(f"\n=== Sanity Check: {DATASET_ID} ({SUBSET}/{SPLIT}) ===\n")

    if RICH:
        console.print("[yellow]Streaming dataset...[/yellow]")

    # decode=False: skip torchcodec, get raw bytes instead
    ds     = load_dataset(DATASET_ID, SUBSET, split=SPLIT, streaming=True, trust_remote_code=True)
    ds     = ds.cast_column("audio", Audio(decode=False))
    sample = next(iter(ds))

    audio_data = sample["audio"]

    # Decode raw bytes with soundfile — no torchcodec needed
    raw = audio_data.get("bytes") or open(audio_data["path"], "rb").read()
    audio_array, sample_rate = sf.read(io.BytesIO(raw))
    audio_array = audio_array.astype(np.float32)

    Path("samples").mkdir(exist_ok=True)
    wav_path = Path("samples") / "librispeech_sanity.wav"
    sf.write(str(wav_path), audio_array, sample_rate)

    if RICH:
        console.print(f"[dim]Audio saved → {wav_path}[/dim]")
    else:
        print(f"Audio saved → {wav_path}")

    model  = load_model()
    result = transcribe(model, str(wav_path))
    print_result(result)
    save_result(result)

    if RICH:
        console.print("\n[green]✓ Done — result appended to asr_results.json[/green]")
    else:
        print("\nDone — result appended to asr_results.json")


if __name__ == "__main__":
    run()
