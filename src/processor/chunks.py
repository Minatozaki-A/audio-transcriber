"""Audio chunks and cumulative timestamp offsets."""

import wave
from dataclasses import dataclass
from pathlib import Path

from errors import AudioConversionError


@dataclass(frozen=True)
class AudioChunk:
    path: Path
    offset_seconds: float


def build_audio_chunks(paths: list[Path]) -> list[AudioChunk]:
    if not paths:
        raise AudioConversionError("No audio chunks were produced")
    chunks: list[AudioChunk] = []
    offset = 0.0
    for path in paths:
        try:
            with wave.open(str(path), "rb") as audio:
                duration = audio.getnframes() / audio.getframerate()
        except (OSError, wave.Error, EOFError) as error:
            raise AudioConversionError(f"Cannot read audio chunk {path}: {error}") from error
        if duration <= 0:
            raise AudioConversionError(f"Empty audio chunk: {path}")
        chunks.append(AudioChunk(path, offset))
        offset += duration
    return chunks
