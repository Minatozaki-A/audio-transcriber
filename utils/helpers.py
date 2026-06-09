import logging
from pathlib import Path
import subprocess
import magic
import uuid

_INPUT_DIR: Path = Path(__file__).parent.parent / "input"
_TEMP_DIR: Path = Path(__file__).parent.parent / "temp"


def _find_audio_file() -> Path | None:
    for file in _INPUT_DIR.iterdir():
        if file.is_file():
            return file
    return None

def generate_name_audio_file(path_dir: Path, label: str|None = None) -> Path:
        unique_id: str = str(uuid.uuid4())[:6]

        file_name: str = f"aud-{unique_id}"

        if label:
            file_name += f"-{label}"

        return path_dir / f"{file_name}.wav"


def convert_to_wav_16_mono() -> Path | None:
    audio_file: Path = _find_audio_file()
    if not audio_file:
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


