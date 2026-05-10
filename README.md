# Whisper ASR Playground

Local ASR testing harness built on [openai/whisper-tiny](https://huggingface.co/openai/whisper-tiny). Transcribe audio files, track metrics across runs, score against reference transcripts, and run automated sanity checks against the `distil-whisper/librispeech_long` dataset — all on CPU, no GPU required.

---

## Features

| Feature | Detail |
|---|---|
| **Local inference** | Whisper-tiny (~75 MB), CPU-only, ~400 MB RAM |
| **Real-Time Factor** | Measures how fast vs. real-time (e.g. `0.11x` = 9× faster than real-time) |
| **Confidence score** | Derived from Whisper's internal `no_speech_prob` per segment |
| **WER scoring** | Paste a reference transcript → get Word Error Rate + accuracy % |
| **JSON history log** | Every run appended to `asr_results.json` for cross-run comparison |
| **Pretty output** | Rich-powered tables and panels (degrades gracefully without `rich`) |
| **Dataset sanity check** | One-command test against `distil-whisper/librispeech_long` |

---

## Setup

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2. Install ffmpeg

| Platform | Command |
|---|---|
| Windows | `winget install ffmpeg` |
| macOS | `brew install ffmpeg` |
| Ubuntu/Debian | `sudo apt install ffmpeg` |

### 3. Verify

```bash
ffmpeg -version
python -c "import whisper; print('ok')"
```

---

## Usage

### Transcribe a file

```bash
python asr_tester.py audio.mp3
```

### Interactive mode (prompts for file)

```bash
python asr_tester.py
```

### View all past results

```bash
python asr_tester.py --history
```

### Run dataset sanity check

Streams one long audio sample from `distil-whisper/librispeech_long` and transcribes it:

```bash
python sanity_check.py
```

---

## Supported formats

`.mp3` `.wav` `.m4a` `.ogg` `.flac` `.mp4` `.webm`

---

## Sample output

```
╭─────────────────────────────────────────────╮
│  ASR Tester — Whisper Tiny (local)          │
│  Model: openai/whisper-tiny  |  ~75 MB      │
╰─────────────────────────────────────────────╯

╭─ Transcription ─────────────────────────────╮
│  Mr. Quilter is the apostle of the middle   │
│  classes, and we are glad to welcome his    │
│  gospel. ...                                │
╰─────────────────────────────────────────────╯

  Language          en
  Audio duration    61.96s
  Processing time   6.51s
  Real-time factor  0.11x  (1x = real-time speed)
  Confidence        97.4%
  Word count        149
  Segments          13
```

---

## Project structure

```
whisper-ASR/
├── asr_tester.py       # main transcription script
├── sanity_check.py     # dataset-driven sanity check
├── requirements.txt    # pip dependencies
├── samples/            # drop audio files here for testing
└── asr_results.json    # auto-generated run history (gitignored)
```

---

## Model options

Edit `MODEL_NAME` in `asr_tester.py` to trade speed for accuracy:

| Model | Size | Relative speed | Best for |
|---|---|---|---|
| `tiny` | 75 MB | fastest | quick tests, prototyping |
| `base` | 145 MB | fast | general use |
| `small` | 483 MB | moderate | better accuracy |
| `medium` | 1.5 GB | slow | high-accuracy transcription |

---

## Dependencies

- [`openai-whisper`](https://github.com/openai/whisper)
- [`rich`](https://github.com/Textualize/rich)
- [`datasets`](https://github.com/huggingface/datasets) *(sanity check only)*
- [`soundfile`](https://github.com/bastibe/python-soundfile) *(sanity check only)*
