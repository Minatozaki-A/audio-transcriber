import logging
import tempfile as tf
from pathlib import Path
import tkinter as tk
from tkinter import filedialog
import subprocess as sp
import magic
import uuid

_TEMP_DIR: Path = Path(tf.gettempdir())

def select_audio_files() -> list[Path]:
    """Open a native file dialog and return the selected audio/video paths."""
    root = tk.Tk()
    root.withdraw()

    file_paths: tuple[str, ...] = filedialog.askopenfilenames(
        parent=root,
        initialdir=str(Path.home()),
        title="Select Audio Files",
        filetypes=[("Audio Files", ["*.mp3", "*.wav", "*.flac", "*.ogg", "*.m4a", "*.aac", "*.opus", "*.wma"]),
        ("Video", ["*.mp4", "*.mkv", "*.avi", "*.mov", "*.webm", "*.flv", "*.wmv", "*.m4v"]),
        ("Todos los archivos", "*.*"),
        ],
    )
    root.destroy()
    return [Path(file_path) for file_path in file_paths]


def _command_ffmpeg(audio_file: Path, output_file: Path):
    """Build the ffmpeg argument list to convert any audio/video to 16 kHz mono WAV (s16)."""
    command: list[str] = [
        "ffmpeg",
        "-y",
        "-i", str(audio_file),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-sample_fmt", "s16",
        str(output_file),
        ]
    return command

def generate_name_audio_file() -> Path:
        """Return a unique temp path of the form <tmpdir>/aud-<9-char-uuid>.wav."""
        unique_id: str = str(uuid.uuid4())[:9]

        file_name: str = f"aud-{unique_id}"

        return _TEMP_DIR / f"{file_name}.wav"


def convert_to_wav_16_mono_v2() -> list[Path]:
    """Prompt the user to pick files, convert each to 16 kHz mono WAV, and return the resulting paths."""
    audio_files_converted: list[Path] = []
    audio_files: list[Path] = select_audio_files()


    for af in audio_files:
        mime: str = magic.from_file(af, mime=True)

        if not (mime.startswith("audio/") or mime.startswith("video/")):
            logging.error("Unsupported file type: %s", mime)
            continue

        if mime in {"audio/wav", "audio/x-wav"}:
            logging.info("File %s already in WAV format, skipping conversion", af.name)
            audio_files_converted.append(af)
            continue

        try:
            new_name: Path = generate_name_audio_file()

            sp.run(_command_ffmpeg(af, new_name), check=True)

            logging.info("Audio file s% | type s% converted to s%",
                            af, mime, new_name, mime)

            audio_files_converted.append(new_name)

        except sp.CalledProcessError as e:
            logging.error("Error converting audio file s%: %s", af.name, e)

    return audio_files_converted