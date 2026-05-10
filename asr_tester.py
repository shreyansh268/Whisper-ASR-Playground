"""
ASR Tester — Local Whisper Tiny
================================
Purpose  : Test transcription quality of whisper-tiny locally
Model    : openai/whisper-tiny  (~75MB, CPU-only, ~400MB RAM)
Usage    : python asr_tester.py [audio_file]
           python asr_tester.py  (interactive mode — prompts for file)

Install deps first:
    pip install openai-whisper rich
    # ffmpeg also required:
    # Ubuntu/Debian : sudo apt install ffmpeg
    # macOS         : brew install ffmpeg
    # Windows       : https://ffmpeg.org/download.html
"""

import sys
import os
import time
import json
from pathlib import Path
from datetime import datetime

# ── Optional rich for pretty output ──────────────────────────────────────────
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich import print as rprint
    RICH = True
    console = Console()
except ImportError:
    RICH = False
    console = None

# ── Whisper ───────────────────────────────────────────────────────────────────
try:
    import whisper
except ImportError:
    print("ERROR: openai-whisper not installed.")
    print("Run:  pip install openai-whisper")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

MODEL_NAME      = "tiny"          # Options: tiny, base, small (tiny = fastest, least RAM)
SUPPORTED_EXTS  = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".mp4", ".webm"}
RESULTS_FILE    = "asr_results.json"   # cumulative test log


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def banner():
    msg = (
        "\n╔══════════════════════════════════════╗\n"
        "║  ASR Tester — Whisper Tiny (local)   ║\n"
        "║  Model: openai/whisper-tiny  ~75 MB  ║\n"
        "╚══════════════════════════════════════╝\n"
    )
    if RICH:
        console.print(Panel.fit("[bold cyan]ASR Tester — Whisper Tiny (local)[/bold cyan]\n"
                                "[dim]Model: openai/whisper-tiny  |  CPU-only  |  ~75 MB[/dim]",
                                border_style="cyan"))
    else:
        print(msg)


def load_model():
    """Load whisper-tiny (downloads once, cached in ~/.cache/whisper)."""
    if RICH:
        console.print("[yellow]Loading whisper-tiny model (first run downloads ~75MB)...[/yellow]")
    else:
        print("Loading model...")

    start = time.time()
    model = whisper.load_model(MODEL_NAME)
    elapsed = time.time() - start

    if RICH:
        console.print(f"[green]✓ Model loaded in {elapsed:.1f}s[/green]")
    else:
        print(f"Model loaded in {elapsed:.1f}s")

    return model


def transcribe(model, audio_path: str) -> dict:
    """Transcribe audio and return a result dict with metrics."""
    path = Path(audio_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {audio_path}")

    if path.suffix.lower() not in SUPPORTED_EXTS:
        raise ValueError(f"Unsupported format: {path.suffix}  |  Supported: {SUPPORTED_EXTS}")

    file_size_mb = path.stat().st_size / (1024 * 1024)

    if RICH:
        console.print(f"\n[cyan]Transcribing:[/cyan] {path.name}  ({file_size_mb:.2f} MB)")
    else:
        print(f"\nTranscribing: {path.name} ({file_size_mb:.2f} MB)")

    start = time.time()

    # Run transcription — verbose=False suppresses internal whisper logs
    result = model.transcribe(str(path), verbose=False)

    elapsed = time.time() - start
    text    = result["text"].strip()
    lang    = result.get("language", "unknown")

    # Rough confidence from segment no_speech_prob (lower = more confident)
    segments = result.get("segments", [])
    avg_no_speech = (
        sum(s.get("no_speech_prob", 0) for s in segments) / len(segments)
        if segments else 0
    )
    confidence_pct = round((1 - avg_no_speech) * 100, 1)

    # Audio duration from segments
    audio_duration = segments[-1]["end"] if segments else 0
    rtf = round(elapsed / audio_duration, 2) if audio_duration > 0 else None  # Real-Time Factor

    return {
        "file"            : path.name,
        "file_size_mb"    : round(file_size_mb, 3),
        "audio_duration_s": round(audio_duration, 2),
        "transcription"   : text,
        "language"        : lang,
        "processing_time_s": round(elapsed, 2),
        "real_time_factor": rtf,        # <1 = faster than real-time, >1 = slower
        "confidence_pct"  : confidence_pct,
        "word_count"      : len(text.split()),
        "segments"        : len(segments),
        "timestamp"       : datetime.now().isoformat(),
        "model"           : MODEL_NAME,
    }


def print_result(res: dict):
    """Pretty-print a single transcription result."""
    if RICH:
        # Transcription panel
        console.print(Panel(
            f"[bold white]{res['transcription']}[/bold white]",
            title="[green]Transcription[/green]",
            border_style="green"
        ))

        # Metrics table
        t = Table(title="Metrics", show_header=False, box=None, padding=(0, 2))
        t.add_column("Key", style="dim")
        t.add_column("Value", style="bold")

        rtf_label = f"{res['real_time_factor']}x" if res['real_time_factor'] else "N/A"
        rtf_color = "green" if (res['real_time_factor'] or 99) <= 2 else "yellow"

        t.add_row("Language",        res["language"])
        t.add_row("Audio duration",  f"{res['audio_duration_s']}s")
        t.add_row("Processing time", f"{res['processing_time_s']}s")
        t.add_row("Real-time factor", f"[{rtf_color}]{rtf_label}[/{rtf_color}]  (1x = real-time speed)")
        t.add_row("Confidence",      f"{res['confidence_pct']}%")
        t.add_row("Word count",      str(res["word_count"]))
        t.add_row("Segments",        str(res["segments"]))

        console.print(t)
    else:
        print("\n--- Transcription ---")
        print(res["transcription"])
        print("\n--- Metrics ---")
        for k, v in res.items():
            if k not in ("transcription", "segments", "timestamp"):
                print(f"  {k}: {v}")


def save_result(res: dict):
    """Append result to cumulative JSON log for comparison over time."""
    history = []
    if Path(RESULTS_FILE).exists():
        with open(RESULTS_FILE) as f:
            history = json.load(f)

    history.append(res)

    with open(RESULTS_FILE, "w") as f:
        json.dump(history, f, indent=2)

    if RICH:
        console.print(f"[dim]Result saved → {RESULTS_FILE}[/dim]")
    else:
        print(f"Result saved → {RESULTS_FILE}")


def show_history():
    """Print summary table of all previous test runs."""
    if not Path(RESULTS_FILE).exists():
        print("No test history yet.")
        return

    with open(RESULTS_FILE) as f:
        history = json.load(f)

    if RICH:
        t = Table(title="ASR Test History", show_lines=True)
        t.add_column("File",        style="cyan", max_width=25)
        t.add_column("Duration",    justify="right")
        t.add_column("Proc. time",  justify="right")
        t.add_column("RTF",         justify="right")
        t.add_column("Confidence",  justify="right")
        t.add_column("Words",       justify="right")
        t.add_column("Lang")
        t.add_column("Timestamp",   style="dim", max_width=20)

        for r in history:
            rtf = f"{r['real_time_factor']}x" if r.get("real_time_factor") else "N/A"
            t.add_row(
                r["file"],
                f"{r['audio_duration_s']}s",
                f"{r['processing_time_s']}s",
                rtf,
                f"{r['confidence_pct']}%",
                str(r["word_count"]),
                r["language"],
                r["timestamp"][:16],
            )
        console.print(t)
    else:
        print(f"\n{'File':<25} {'RTF':>6} {'Conf':>6} {'Words':>6} {'Lang':>6}")
        print("-" * 60)
        for r in history:
            rtf = f"{r['real_time_factor']}x" if r.get("real_time_factor") else "N/A"
            print(f"{r['file']:<25} {rtf:>6} {r['confidence_pct']:>5}% {r['word_count']:>6} {r['language']:>6}")


def get_reference_and_score(transcription: str) -> dict | None:
    """
    Optional: compare transcription against a known reference text.
    Returns WER (Word Error Rate) — lower = better.
    """
    print("\nDo you have a reference transcript to compare against? (y/n): ", end="")
    choice = input().strip().lower()

    if choice != "y":
        return None

    print("Paste the reference text (press Enter twice when done):")
    lines = []
    while True:
        line = input()
        if line == "" and lines and lines[-1] == "":
            break
        lines.append(line)

    reference = " ".join(lines).strip()

    # Simple WER calculation
    ref_words  = reference.lower().split()
    hyp_words  = transcription.lower().split()

    # Levenshtein distance at word level
    def wer(ref, hyp):
        n, m = len(ref), len(hyp)
        dp = [[0] * (m + 1) for _ in range(n + 1)]
        for i in range(n + 1): dp[i][0] = i
        for j in range(m + 1): dp[0][j] = j
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                dp[i][j] = (
                    dp[i-1][j-1] if ref[i-1] == hyp[j-1]
                    else 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
                )
        return dp[n][m] / max(len(ref), 1)

    score = round(wer(ref_words, hyp_words) * 100, 2)
    accuracy = round(100 - score, 2)

    if RICH:
        color = "green" if accuracy >= 90 else "yellow" if accuracy >= 75 else "red"
        console.print(f"\n[bold]WER (Word Error Rate):[/bold] {score}%")
        console.print(f"[bold]Accuracy:[/bold] [{color}]{accuracy}%[/{color}]")
    else:
        print(f"\nWER: {score}%  |  Accuracy: {accuracy}%")

    return {"wer_pct": score, "accuracy_pct": accuracy, "reference": reference}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    banner()

    # ── Special commands ──────────────────────────────────────────────────────
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == "--history":
            show_history()
            return
        if arg == "--help":
            print(__doc__)
            return
        audio_path = arg
    else:
        print("\nCommands:")
        print("  python asr_tester.py <audio_file>   — transcribe a file")
        print("  python asr_tester.py --history       — show all past results")
        print("  python asr_tester.py --help          — show help")
        print()
        audio_path = input("Enter audio file path (or drag & drop): ").strip().strip("'\"")

    # ── Load model ────────────────────────────────────────────────────────────
    model = load_model()

    # ── Transcribe ────────────────────────────────────────────────────────────
    try:
        result = transcribe(model, audio_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    print_result(result)

    # ── Optional WER scoring ──────────────────────────────────────────────────
    wer_data = get_reference_and_score(result["transcription"])
    if wer_data:
        result.update(wer_data)

    # ── Save to history ───────────────────────────────────────────────────────
    save_result(result)

    # ── Offer to test another ─────────────────────────────────────────────────
    print("\nTest another file? (y/n): ", end="")
    again = input().strip().lower()
    if again == "y":
        audio_path = input("File path: ").strip().strip("'\"")
        try:
            result = transcribe(model, audio_path)
            print_result(result)
            save_result(result)
        except Exception as e:
            print(f"ERROR: {e}")


if __name__ == "__main__":
    main()
