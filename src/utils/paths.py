"""Generated audio paths and the shared output directory."""

import uuid
from pathlib import Path


_ECHOBEAK_DIR: Path = Path.home() / "EchoBeak-Transcription"


def create_temp_audio_path(temp_dir: Path | str) -> Path:
    random_number = uuid.uuid4()
    return Path(temp_dir) / f"temp-{random_number}.wav"


def get_output_directory() -> Path:
    _ECHOBEAK_DIR.mkdir(parents=True, exist_ok=True)
    return _ECHOBEAK_DIR
