import logging
import wave
import subprocess as sp
import magic
import uuid
from datetime import date
from pathlib import Path


_ECHOBEAK_DIR: Path = Path.home() / "EchoBeak"


def _command_ffmpeg(audio_file: Path, output_file: Path):
    """Build the ffmpeg argument list to convert any audio/video to 16 kHz mono WAV (s16)."""
    command: list[str] = [
        "ffmpeg",
        "-y",
        "-i", str(audio_file),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-sample_fmt", "s16",
        str(output_file),
        ]
    return command

def _is_target_wav_format_stdlib(file_path: Path) -> bool:
    try:

        with wave.open(str(file_path), "rb") as wf:
            return (
                    wf.getframerate() == 16000
                    and wf.getnchannels() == 1
                    and wf.getsampwidth() == 2 # 2 bytes = 16 bits = s16
            )

    except (wave.Error, IOError) as e:
        logging.error("Cannot read WAV header of %s: %s", file_path.name, e)
        return False


def _unique_path(base_path: Path, max_attempts: int = 1000) -> Path:
    """Genera una ruta única añadiendo un sufijo numérico para evitar colisiones.

    Returns:
        Path | None: La ruta única si se encuentra,
    """
    if not base_path.exists():
        return base_path
    stem, suffix = base_path.stem, base_path.suffix
    for n in range(1, max_attempts + 1):
        candidate: Path = base_path.parent / f"{stem}-({n}){suffix}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"could not find unique path for {base_path} after {max_attempts} attempts")


def create_temp_audio_path(temp_dir: str) -> Path:
    random_number = uuid.uuid4()
    return Path(f"{temp_dir}/temp-{random_number}.wav")

def create_name_trans_path() -> Path:
    new_name = f"trans-{date.today():%Y-%m-%d}.md"

    _ECHOBEAK_DIR.mkdir(parents=True, exist_ok=True)
    final_name: Path = _ECHOBEAK_DIR / new_name

    return _unique_path(final_name)



def _split_wav_5_minutes(audio_file: Path, output_file: Path) -> list[Path]:
    """Split a normalized WAV into consecutive five-minute files with ffmpeg."""
    pattern = output_file.with_name(f"{output_file.stem}-part-%03d.wav")
    sp.run(
        ["ffmpeg", "-y", "-i", str(audio_file), "-af", "asetnsamples=n=16000:p=0", "-f", "segment",
         "-segment_time", "300", "-reset_timestamps", "1", str(pattern)],
        check=True,
    )
    return sorted(output_file.parent.glob(f"{output_file.stem}-part-*.wav"))


def convert_to_wav_16_mono(selected_audio_file: Path, temp_dir: str) -> list[Path]:
    """Return a target WAV directly, or convert and split longer audio."""
    try:
        mime: str = magic.from_file(selected_audio_file, mime=True)
    except OSError:
        logging.error("Cannot read file %s", selected_audio_file.name)
        return []
    except magic.MagicException:
        logging.exception("Cannot detect MIME type for file %s", selected_audio_file)
        return []

    if not (mime.startswith("audio/") or mime.startswith("video/")):
        logging.error("Unsupported file type: %s (%s)", mime, selected_audio_file.name)
        return []

    already_target_wav = (
        mime in {"audio/wav", "audio/x-wav"}
        and _is_target_wav_format_stdlib(selected_audio_file)
    )
    output_file = create_temp_audio_path(temp_dir)
    try:
        audio_file = selected_audio_file if already_target_wav else output_file
        if not already_target_wav:
            sp.run(_command_ffmpeg(selected_audio_file, output_file), check=True)

        with wave.open(str(audio_file), "rb") as wf:
            duration_seconds = wf.getnframes() / wf.getframerate()
        if duration_seconds <= 300:
            return [audio_file]

        try:
            parts = _split_wav_5_minutes(audio_file, output_file)
            if not parts:
                raise OSError("ffmpeg did not create any audio segments")
            return parts
        finally:
            output_file.unlink(missing_ok=True)
    except sp.CalledProcessError as e:
        logging.error("Error processing audio file %s: %s", selected_audio_file.name, e)
    except FileNotFoundError:
        logging.error("ffmpeg not found in PATH — aborting conversion")
        raise
    except (OSError, wave.Error) as e:
        logging.error("I/O error processing %s: %s", selected_audio_file.name, e)

    output_file.unlink(missing_ok=True)
    for part in output_file.parent.glob(f"{output_file.stem}-part-*.wav"):
        part.unlink(missing_ok=True)
    return []
