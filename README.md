# audio-transcriber

Transcribe audio and video files locally using [faster-whisper](https://github.com/SYSTRAN/faster-whisper). The pipeline converts any audio/video format to WAV, applies non-stationary noise reduction, and outputs timestamped transcription segments.

## Pipeline

```
input/<file>  →  ffmpeg (WAV 16kHz mono)  →  noise reduction  →  faster-whisper  →  stdout
```

1. **Conversion** — detects any audio/video file in `input/` and converts it to 16 kHz mono WAV via `ffmpeg`.
2. **Noise reduction** — applies non-stationary noise reduction with `noisereduce`.
3. **Transcription** — runs Whisper (`small` model, CPU + INT8) and prints each segment with timestamps.
4. **Cleanup** — removes all temporary files from `temp/`.

## Requirements

- Python 3.13+
- [uv](https://docs.astral.sh/uv/)
- `ffmpeg`

## Setup

```bash
uv sync
```

## Usage

1. Place an audio or video file inside the `input/` directory.
2. Export the required library paths (required for CUDA/cuDNN support):

```bash
export LD_LIBRARY_PATH="$PWD/.venv/lib/python3.13/site-packages/nvidia/cublas/lib:$PWD/.venv/lib/python3.13/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH"
```

3. Run:

```bash
uv run main.py
```

Output example:

```
2026-06-12 10:00:00 [INFO] Detected language 'es' with probability 0.987431
[0.00s -> 3.20s] Hola, esto es una prueba de transcripción.
[3.20s -> 6.50s] El sistema funciona correctamente.
```

## Project structure

```
audio-transcriber/
├── input/              # Place source audio/video files here
├── temp/               # Intermediate WAV files (auto-cleaned)
├── processor/
│   ├── cleaner.py      # Noise reduction and normalization
│   └── transcriber.py  # faster-whisper transcription
├── utils/
│   └── helpers.py      # ffmpeg conversion, file detection, cleanup
├── main.py
└── pyproject.toml
```

## GPU acceleration

By default the transcriber runs on CPU with INT8 quantization. To use a CUDA GPU, edit `processor/transcriber.py:18`:

```python
# FP16
model = WhisperModel(_MODEL_SIZE, device="cuda", compute_type="float16")

# INT8 mixed (lower VRAM)
model = WhisperModel(_MODEL_SIZE, device="cuda", compute_type="int8_float16")
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
