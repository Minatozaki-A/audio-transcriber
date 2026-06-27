# audio-transcriber

Transcribe audio and video files locally using [faster-whisper](https://github.com/SYSTRAN/faster-whisper). The pipeline converts any audio/video format to WAV, applies non-stationary noise reduction, and outputs timestamped transcription segments. Multiple files can be selected and processed in a single run.

## Pipeline

```
file dialog → ffmpeg (WAV 16kHz mono) → noise reduction → faster-whisper → stdout
```

1. **Selection** — a native file dialog opens so you can pick one or more audio/video files.
2. **Conversion** — each file is converted to 16 kHz mono WAV via `ffmpeg` and written to the system temp directory.
3. **Noise reduction** — applies non-stationary noise reduction with `noisereduce`.
4. **Transcription** — runs Whisper (`small` model, CPU + INT8) and prints each segment with timestamps.

## Requirements

- Python 3.13+
- [uv](https://docs.astral.sh/uv/)
- `ffmpeg`

## Setup

```bash
uv sync
```

## Usage

1. Export the required library paths (needed for CUDA/cuDNN support):

```bash
export LD_LIBRARY_PATH="$PWD/.venv/lib/python3.13/site-packages/nvidia/cublas/lib:$PWD/.venv/lib/python3.13/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH"
```

2. Run:

```bash
uv run src/main.py
```

3. A file dialog will open — select one or more audio or video files and click **Open**.

Output example:

```
2026-06-12 10:00:00 [INFO] Transcribing: aud-3f2a1b9.wav
[0.00s -> 3.20s] Hola, esto es una prueba de transcripción.
[3.20s -> 6.50s] El sistema funciona correctamente.
```

## Project structure

```
audio-transcriber/
├── src/
│   ├── processor/
│   │   ├── cleaner.py      # Noise reduction
│   │   └── transcriber.py  # faster-whisper transcription
│   ├── utils/
│   │   └── helpers.py      # File dialog, ffmpeg conversion
│   └── main.py
├── pyproject.toml
└── uv.lock
```

## GPU acceleration

By default the transcriber runs on CPU with INT8 quantization. To use a CUDA GPU, edit `src/main.py:43`:

```python
# FP16
with whisper_model("medium", device="cuda", compute_type="float16") as model:

# INT8 mixed (lower VRAM)
with whisper_model("medium", device="cuda", compute_type="int8_float16") as model:
```

## Dependencies

| Package | Purpose |
|---|---|
| `faster-whisper` | CTranslate2-based Whisper inference |
| `noisereduce` | Non-stationary noise reduction |
| `librosa` | Audio loading and normalization |
| `soundfile` | WAV read/write |
| `python-magic` | MIME type detection |
| `torch` | CUDA memory management |
