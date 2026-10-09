import argparse
import logging
import sys
import tempfile
from collections.abc import Iterator, Sequence
from pathlib import Path

from errors import AudioConversionError, NoiseReductionError, TranscriptionError
from processor.cleaner import reduce_noise
from processor.transcriber import load_whisper_model
from utils.helpers import (
    AudioChunk,
    build_audio_chunks,
    convert_to_wav_16_mono,
    create_name_trans_path,
    publish_transcript,
    validate_transcript_name,
)


def _transcript_lines(
    chunks: list[AudioChunk], temp_dir: Path, language: str | None
) -> Iterator[str]:
    try:
        with load_whisper_model() as model:
            for chunk in chunks:
                cleaned_path = reduce_noise(chunk.path, temp_dir)
                logging.info("Transcribing: %s", chunk.path.name)
                try:
                    segments, _ = model.transcribe(
                        str(cleaned_path), beam_size=5, language=language
                    )
                    for segment in segments:
                        start = chunk.offset_seconds + segment.start
                        end = chunk.offset_seconds + segment.end
                        yield f"[{start:.2f}s -> {end:.2f}s] {segment.text}\n"
                except Exception as error:
                    raise TranscriptionError(
                        f"Cannot transcribe {chunk.path}: {error}"
                    ) from error
    except (NoiseReductionError, TranscriptionError):
        raise
    except Exception as error:
        raise TranscriptionError(f"Whisper model failed: {error}") from error


def transcribe_file(
    input_path: Path, output_path: Path, *, language: str | None = None
) -> Path:
    """Transcribe audio with global timestamps and publish only after success."""
    input_path = input_path.expanduser()
    output_path = output_path.expanduser()
    if not input_path.is_file():
        raise AudioConversionError(f"Input file does not exist: {input_path}")
    validate_transcript_name(output_path.name)

    with tempfile.TemporaryDirectory() as directory:
        temp_dir = Path(directory)
        paths = convert_to_wav_16_mono(input_path, temp_dir)
        chunks = build_audio_chunks(paths)
        draft_path = temp_dir / "transcript.md"
        lines = _transcript_lines(chunks, temp_dir, language)
        try:
            with draft_path.open("w", encoding="utf-8") as output:
                for line in lines:
                    output.write(line)
        finally:
            # Close the model context even if writing the draft fails.
            lines.close()
        return publish_transcript(draft_path, output_path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Transcribe un archivo de audio o video.")
    parser.add_argument("input_path", type=Path, help="Ruta del archivo de audio o video")
    parser.add_argument("name", help="Nombre del archivo Markdown de transcripción")
    args = parser.parse_args(argv)
    if not args.input_path.expanduser().is_file():
        parser.error(f"No existe el archivo de entrada: {args.input_path}")
    try:
        validate_transcript_name(args.name)
        output_path = transcribe_file(args.input_path, create_name_trans_path(args.name))
    except (
        AudioConversionError, NoiseReductionError, TranscriptionError,
        OSError, ValueError, RuntimeError,
    ):
        logging.exception("Transcription failed")
        return 1
    logging.info("Transcription saved to %s", output_path)
    return 0


def main_v2() -> None:
    """Compatibility entry point for existing callers."""
    sys.exit(main())


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    sys.exit(main())
