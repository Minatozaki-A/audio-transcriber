import logging
import tempfile as tf
from pathlib import Path
import tkinter as tk
from tkinter import filedialog
import subprocess as sp
import magic
import uuid

_INPUT_DIR: Path = Path(__file__).parents[2]/"input"
_TEMP_DIR: Path = Path(tf.gettempdir())

def find_audio_file() -> Path | None:
    if not _INPUT_DIR.exists():
        logging.warning(f"Input directory does not exist: {_INPUT_DIR}")
        return None

    for file in _INPUT_DIR.iterdir():
        if not file.is_file():
            continue
        mime = magic.from_file(file, mime=True)
        if mime.startswith("audio/") or mime.startswith("video/"):
            logging.info(f"Found audio file: {file} - type {mime}")
            return file

    logging.warning("No audio file found in input directory")
    return None

def select_audio_files() -> list[str]:
    root = tk.Tk()
    root.geometry("800x600")
    root.withdraw()

    file_paths: tuple[str, ...] = filedialog.askopenfilenames(
        parent=root,
        initialdir=str(Path.home()),
        title="Select Audio Files",
        filetypes=[("Audio Files", "*.*"),
        ("Video", "*.*"),
        ("Todos los archivos", "*.*"),
        ],
    )
    root.destroy()
# general copia de los archivos y un directorio en tmp
    return [Path(file_path) for file_path in file_paths]


def generate_name_audio_file(path_dir: Path) -> Path:
        unique_id: str = str(uuid.uuid4())[:9]

        file_name: str = f"aud-{unique_id}"

        return path_dir / f"{file_name}.wav"

"""def remove_temp_files() -> None:
    for file in _TEMP_DIR.iterdir():
        if file.is_file():
            try:
                file.unlink()
                logging.info(f"Removed temporary file: {file.stem}")
            except FileNotFoundError:
                logging.warning(f"File not found: {file.stem}")
            except PermissionError:
                logging.error(f"Permission denied to remove file: {file.stem}")"""


def convert_to_wav_16_mono() -> Path | None:
    audio_file: Path | None = find_audio_file()

    if not audio_file:
        logging.error("file not found")
        return None

    mime: str = magic.from_file(audio_file, mime=True)

    """if mime in {"audio/wav", "audio/x-wav"}:
        logging.info(f"Audio file {audio_file} - type {magic.from_file(audio_file, mime=True)} already in WAV format")
        return audio_file"""

    if not mime.startswith("audio/") or mime.startswith("video/"):
        logging.error(f"Unsupported file type: %s", mime)
        return None

    new_name: Path = generate_name_audio_file(_TEMP_DIR)
    command: list[str] = [
        "ffmpeg",
        "-y",
        "-i", str(audio_file),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-sample_fmt", "s16",
        str(new_name),
        ]

    try:
        subprocess.run(command, check=True)

        logging.info(f"Audio file {audio_file.stem} - type {magic.from_file(audio_file, mime=True)}"
                    f" converted to {new_name.stem} - type {magic.from_file(new_name, mime=True)}")

        return new_name

    except subprocess.CalledProcessError as e:
        logging.error(f"Error converting audio file {audio_file.name}: {e}")


