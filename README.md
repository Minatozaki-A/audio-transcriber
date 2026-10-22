# audio-transcriber

Transcribe one audio or video file locally with [faster-whisper](https://github.com/SYSTRAN/faster-whisper). The pipeline checks the MIME type, converts the file to 16 kHz mono 16-bit WAV, splits audio longer than twenty minutes, optionally applies non-stationary noise reduction, and writes a timestamped Markdown transcript to `~/EchoBeak`.

## Pipeline

```
input path + transcript name
  → MIME check
  → ffmpeg (WAV 16 kHz mono s16), or the file itself when it already matches
  → twenty-minute split when the audio is longer than 1200 seconds
  → optional noise reduction (--noise-reduce)
  → faster-whisper
  → ~/EchoBeak/<name>.md
```

1. **Input** — the command takes one audio or video path and a transcript file name.
2. **Conversion** — `python-magic` requires an `audio/*` or `video/*` MIME type. A WAV that is already 16 kHz, mono, and 16-bit skips conversion. Every other accepted file is converted with `ffmpeg` (`-vn -ac 1 -ar 16000 -sample_fmt s16`).
3. **Splitting** — audio longer than twenty minutes is cut into consecutive twenty-minute WAV parts. Timestamps in the transcript stay continuous across parts.
4. **Noise reduction** — disabled by default. With `--noise-reduce`, each part is processed with non-stationary `noisereduce` and written to a temporary WAV.
5. **Transcription** — Whisper runs with beam size 5. Use `--language es` to specify the audio language for every chunk. When omitted, the model detects the language (`language=None`). `transcribe_file()` also accepts a language code when called from Python. The model is loaded before conversion to validate explicit codes against `model.supported_languages`. Unsupported codes stop the run with status 1 and a message listing the supported codes, before conversion or noise reduction. Language rejections from the library are also reported without a traceback.
6. **Publish** — segments are written to a temporary draft and copied to `~/EchoBeak` only after the whole file succeeds. Each segment is one line: `[start -> end] text`.

Each processed temporary WAV is deleted with `Path.unlink()`. Remaining temporary files are removed by `tempfile.TemporaryDirectory` when the run ends. The source file stays in place.

## Requirements

- Python 3.13+
- [uv](https://docs.astral.sh/uv/)
- `ffmpeg` on `PATH`
- the system `libmagic` library used by `python-magic`

## Setup

```bash
uv sync
```

For a CUDA run, expose the cuBLAS and cuDNN libraries shipped in the virtual environment:

```bash
export LD_LIBRARY_PATH="$PWD/.venv/lib/python3.13/site-packages/nvidia/cublas/lib:$PWD/.venv/lib/python3.13/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH"
```

## Usage

```bash
uv run src/main.py ~/recordings/meeting.mp4 meeting
```

To specify Spanish instead of automatic language detection:

```bash
uv run src/main.py ~/recordings/meeting.mp4 meeting --language es
```

To enable noise reduction:

```bash
uv run src/main.py ~/recordings/meeting.mp4 meeting --noise-reduce
```

Arguments:

| Argument | Meaning |
|---|---|
| `input_path` | Path to one audio or video file |
| `--noise-reduce` | Enable non-stationary noise reduction (disabled by default). |
| `--language CODE` | Audio language code, such as `es` (Spanish) or `en` (English). Omit for automatic detection. This selects the transcription language; it does not translate the audio. |
| `name` | Transcript file name. It must be a single file name: empty names, `.`, `..`, and names containing `/` or `\` are rejected. `meeting` and `meeting.md` are both saved as `meeting.md` |

The transcript is saved as `~/EchoBeak/meeting.md`. The directory is created when needed. When that name is already taken, the new file is `meeting-(1).md`, then `meeting-(2).md`, and so on.

Logs go to stderr. The Markdown file holds the transcript:

```
2026-10-20 10:00:00,000 [INFO] Loading model in CPU with INT8
2026-10-20 10:00:01,000 [INFO] Model loaded successfully: small (cpu, int8)
2026-10-20 10:00:01,100 [INFO] Transcribing: temp-3f2a1b9c-4d5e-6789-abcd-ef0123456789.wav
2026-10-20 10:00:12,000 [INFO] Transcription saved to /home/user/EchoBeak/meeting.md
```

```
[0.00s -> 3.20s] Hola, esto es una prueba de transcripción.
[3.20s -> 6.50s] El sistema funciona correctamente.
```

A failed conversion, noise-reduction, or transcription run logs the exception and exits with status 1. A previous transcript at the destination is left as it was.

Run the command once per file to transcribe several recordings.

## Project structure

```
audio-transcriber/
├── src/
│   ├── cli/                    # empty
│   ├── errors.py               # Conversion, noise-reduction, and transcription errors
│   ├── main.py                 # CLI and pipeline
│   ├── processor/
│   │   ├── cleaner.py          # Non-stationary noise reduction
│   │   ├── model.py            # empty
│   │   ├── runtime.py          # empty
│   │   └── transcriber.py      # Model load, GPU fallback, CUDA cleanup
│   └── utils/
│       └── helpers.py          # MIME check, ffmpeg conversion, splitting, transcript paths
├── tests/
│   ├── conftest.py
│   ├── test_cleaner.py
│   ├── test_helpers.py
│   ├── test_pipeline.py
│   └── test_transcriber.py
├── pyproject.toml
├── uv.lock
├── LICENSE
└── README.md
```

## Model selection

`load_whisper_model()` chooses the model at runtime:

- When CTranslate2 reports a CUDA device, it loads `medium` on `cuda` with `float16`.
- When that load fails, when no CUDA device is present, or when the device check itself fails, it loads `small` on `cpu` with `int8`.

An error while transcribing stops the run. The CPU model is the fallback for loading, and the GPU model is unloaded with a CUDA cache clear when it was loaded on `cuda`.

## Tests

`pytest` is configured in `pyproject.toml` (`pythonpath = ["src"]`). The command below supplies it for that run:

```bash
uv run --with pytest pytest
uv run python -m compileall src
```

## Dependencies

| Package | Purpose |
|---|---|
| `faster-whisper` | CTranslate2-based Whisper inference |
| `ctranslate2` | CUDA device detection |
| `noisereduce` | Non-stationary noise reduction |
| `soundfile` | WAV read/write for noise reduction |
| `python-magic` | MIME type detection |
| `torch` | CUDA memory cleanup after a GPU model unloads |
| `nvidia-cublas-cu12` | Bundled cuBLAS for CUDA runs |
| `nvidia-cudnn-cu12` | Bundled cuDNN for CUDA runs |
