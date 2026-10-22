import wave
from pathlib import Path
from collections.abc import Callable

import pytest


@pytest.fixture
def make_wav() -> Callable[..., Path]:
    def write(path: Path, seconds: float = 1.0, rate: int = 16000) -> Path:
        with wave.open(str(path), "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(rate)
            audio.writeframes(b"\x00\x00" * round(seconds * rate))
        return path
    return write
