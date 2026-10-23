import argparse
import logging
import sys
import tempfile
from collections.abc import Sequence
from contextlib import ExitStack
from pathlib import Path

from errors import (
    AudioConversionError, NoiseReductionError, TranscriptionError, UnsupportedLanguageError,
)
from models.model import load_whisper_model
from processor.chunks import build_audio_chunks
from processor.converter import convert_to_wav_16_mono
from processor.transcriber import transcript_lines
from utils.paths import get_output_directory
from utils.transcripts import (
    create_name_trans_path,
    publish_transcript,
    validate_transcript_name,
)


def transcribe_file(
    input_path: Path, output_path: Path, *, language: str | None = None,
    noise_reduce: bool = False, keep_wave: bool = False,
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
            audio_dir = get_output_directory() if keep_wave else temp_dir
            if keep_wave:
                logging.info("Keeping generated WAV files in %s", audio_dir)
            paths = convert_to_wav_16_mono(input_path, audio_dir, keep_wave=keep_wave)
            chunks = build_audio_chunks(paths)
            draft_path = temp_dir / "transcript.md"
            lines = transcript_lines(
                model, chunks, audio_dir, language,
                noise_reduce=noise_reduce, keep_wave=keep_wave,
            )
            try:
                with draft_path.open("w", encoding="utf-8") as output:
                    for line in lines:
                        output.write(line)
            finally:
                # Close the generator to clean the current chunk if writing fails.
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
    parser.add_argument(
        "--keep-wave",
        action="store_true",
        help="Conserva los WAV generados en ~/EchoBeak-Transcription (se eliminan por defecto)",
    )
    args = parser.parse_args(argv)
    if not args.input_path.expanduser().is_file():
        parser.error(f"No existe el archivo de entrada: {args.input_path}")
    try:
        validate_transcript_name(args.name)
        output_path = transcribe_file(
            args.input_path, create_name_trans_path(args.name), language=args.language,
            noise_reduce=args.noise_reduce, keep_wave=args.keep_wave,
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
