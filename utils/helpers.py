import logging
from pathlib import Path
import datetime
import string
import random
import subprocess
import magic

_INPUT_DIR: Path = Path(__file__).parent.parent / "input"
_TEMP_DIR: Path = Path(__file__).parent.parent / "temp"

logging.basicConfig(level=logging.INFO)


def _get_audio_file_input_dir() -> Path:
    files: list[Path] = [f for f in _INPUT_DIR.iterdir() if f.is_file()]
    if not files:
        raise FileNotFoundError(f"No audio file found in {_INPUT_DIR}")
    if len(files) > 1:
        raise ValueError(f"Multiple files found in {_INPUT_DIR}: {[f.name for f in files]}")
    return files[0]


def _generate_name_audio_file() -> Path:
        creation_date: str = datetime.datetime.now().strftime("%Y%m%d")

        subname_video: str = "".join([random.choice(string.ascii_lowercase + string.digits) for _ in range(6)])

        new_name: str = f"aud-{creation_date}-{subname_video}.wav"
        path_output: Path = _TEMP_DIR / new_name
        if path_output.exists():
            return _generate_name_audio_file()
        return path_output


def convert_to_wav_16_mono():
    audio_file: Path = _get_audio_file_input_dir()
    new_name: Path = _generate_name_audio_file()
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

    subprocess.run(command, check=True)
    logging.info(f"Audio file {audio_file.name} - type {magic.from_file(audio_file, mime=True)}"
                f" converted to {new_name} - type {magic.from_file(new_name, mime=True)}")



def convert_video_to_wav_15_mono():
    audio_file: Path = _get_audio_file_input_dir()
    new_name: Path = _generate_name_audio_file()
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
    subprocess.run(command, check=True)
    logging.info(f"Audio file {audio_file.stem} - type {magic.from_file(audio_file, mime=True)}"
                f" converted to {new_name.stem} - type {magic.from_file(new_name, mime=True)}")
