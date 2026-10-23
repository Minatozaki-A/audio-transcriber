"""Audio chunk transcription and timestamped Markdown lines."""

import logging
from collections.abc import Iterator
from pathlib import Path

from faster_whisper import WhisperModel

from errors import TranscriptionError, UnsupportedLanguageError
from processor.cleaner import reduce_noise
from processor.chunks import AudioChunk


def transcript_lines(
    model: WhisperModel, chunks: list[AudioChunk], temp_dir: Path, language: str | None,
    *, noise_reduce: bool = False, keep_wave: bool = False,
) -> Iterator[str]:
    """Yield Markdown segments with global timestamps and clean processed audio."""
    for chunk in chunks:
        audio_path = chunk.path
        try:
            if noise_reduce:
                audio_path = reduce_noise(chunk.path, temp_dir)
            logging.info("Transcribing: %s", chunk.path.name)
            try:
                segments, _ = model.transcribe(
                    str(audio_path), beam_size=5, language=language
                )
                for segment in segments:
                    start = chunk.offset_seconds + segment.start
                    end = chunk.offset_seconds + segment.end
                    yield f"[{start:.2f}s -> {end:.2f}s] {segment.text}\n"
            except ValueError as error:
                if "is not a valid language code" in str(error):
                    raise UnsupportedLanguageError(
                        f"Código de idioma no aceptado por la biblioteca: {language!r}. {error}"
                    ) from error
                raise TranscriptionError(
                    f"Cannot transcribe {chunk.path}: {error}"
                ) from error
            except Exception as error:
                raise TranscriptionError(
                    f"Cannot transcribe {chunk.path}: {error}"
                ) from error
        finally:
            for path in {chunk.path, audio_path}:
                if not keep_wave and path.resolve().is_relative_to(temp_dir.resolve()):
                    try:
                        path.unlink(missing_ok=True)
                    except OSError:
                        logging.exception("Cannot remove temporary audio %s", path)
