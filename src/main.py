import argparse
import logging
import sys
import tempfile
from collections.abc import Iterator, Sequence
from contextlib import ExitStack
from pathlib import Path

from faster_whisper import WhisperModel

from errors import (
    AudioConversionError, NoiseReductionError, TranscriptionError, UnsupportedLanguageError,
)
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
    model: WhisperModel, chunks: list[AudioChunk], temp_dir: Path, language: str | None,
    *, noise_reduce: bool = False,
) -> Iterator[str]:
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
                if path.resolve().is_relative_to(temp_dir.resolve()):
                    try:
                        path.unlink(missing_ok=True)
                    except OSError:
                        logging.exception("Cannot remove temporary audio %s", path)


def transcribe_file(
    input_path: Path, output_path: Path, *, language: str | None = None,
    noise_reduce: bool = False,
) -> Path:
    """Transcribe audio with global timestamps and publish only after success."""
    input_path = input_path.expanduser()
    output_path = output_path.expanduser()
    if not input_path.is_file():
        raise AudioConversionError(f"Input file does not exist: {input_path}")
    validate_transcript_name(output_path.name)

    with ExitStack() as stack:
        try:
            model = stack.enter_context(load_whisper_model())
        except Exception as error:
            raise TranscriptionError(f"Whisper model failed: {error}") from error
        if language is not None:
            supported_languages = model.supported_languages
            if language not in supported_languages:
                raise UnsupportedLanguageError(
                    f"Código de idioma no aceptado: {language!r}. "
                    f"Idiomas admitidos por el modelo: {', '.join(supported_languages)}"
                )

        with tempfile.TemporaryDirectory() as directory:
            temp_dir = Path(directory)
            paths = convert_to_wav_16_mono(input_path, temp_dir)
            chunks = build_audio_chunks(paths)
            draft_path = temp_dir / "transcript.md"
            lines = _transcript_lines(model, chunks, temp_dir, language, noise_reduce=noise_reduce)
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
    parser.add_argument(
        "--language",
        metavar="CODE",
        help="Código del idioma del audio (por ejemplo, es o en); por defecto se detecta automáticamente",
    )
    parser.add_argument(
        "--noise-reduce",
        action="store_true",
        help="Aplica reducción de ruido antes de transcribir (desactivada por defecto)",
    )
    args = parser.parse_args(argv)
    if not args.input_path.expanduser().is_file():
        parser.error(f"No existe el archivo de entrada: {args.input_path}")
    try:
        validate_transcript_name(args.name)
        output_path = transcribe_file(
            args.input_path, create_name_trans_path(args.name), language=args.language,
            noise_reduce=args.noise_reduce,
        )
    except UnsupportedLanguageError as error:
        logging.error("%s", error)
        return 1
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
