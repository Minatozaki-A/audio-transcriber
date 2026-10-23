"""Audio/video validation, WAV conversion, and twenty-minute splitting."""

import logging
import subprocess as sp
import wave
from pathlib import Path

import magic

from errors import AudioConversionError
from utils.paths import create_temp_audio_path


def _command_ffmpeg(audio_file: Path, output_file: Path) -> list[str]:
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

    except (wave.Error, IOError, EOFError) as e:
        logging.error("Cannot read WAV header of %s: %s", file_path.name, e)
        return False


def run_ffmpeg(command: list[str]) -> None:
    try:
        sp.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as error:
        raise AudioConversionError("ffmpeg not found in PATH") from error
    except sp.CalledProcessError as error:
        detail = (error.stderr or "").strip()
        raise AudioConversionError(f"ffmpeg failed: {detail or error}") from error
    except OSError as error:
        raise AudioConversionError(f"Cannot execute ffmpeg: {error}") from error


def _split_wav_20_minutes(audio_file: Path, output_file: Path) -> list[Path]:
    """Split a normalized WAV into consecutive twenty-minute files with ffmpeg."""
    pattern = output_file.with_name(f"{output_file.stem}-part-%03d.wav")
    run_ffmpeg(
        ["ffmpeg", "-y", "-i", str(audio_file), "-af", "asetnsamples=n=16000:p=0", "-f", "segment",
         "-segment_time", "1200", "-reset_timestamps", "1", str(pattern)],
    )
    return sorted(output_file.parent.glob(f"{output_file.stem}-part-*.wav"))


def convert_to_wav_16_mono(
    selected_audio_file: Path, temp_dir: Path | str, *, keep_wave: bool = False,
) -> list[Path]:
    """Return a target WAV directly, or convert and split longer audio."""
    try:
        mime: str = magic.from_file(selected_audio_file, mime=True)
    except (OSError, magic.MagicException) as error:
        raise AudioConversionError(f"Cannot detect MIME type for {selected_audio_file}: {error}") from error

    if not (mime.startswith("audio/") or mime.startswith("video/")):
        raise AudioConversionError(f"Unsupported file type: {mime} ({selected_audio_file})")

    already_target_wav = (
        mime in {"audio/wav", "audio/x-wav"}
        and _is_target_wav_format_stdlib(selected_audio_file)
    )
    output_file = create_temp_audio_path(temp_dir)
    try:
        audio_file = selected_audio_file if already_target_wav else output_file
        if not already_target_wav:
            run_ffmpeg(_command_ffmpeg(selected_audio_file, output_file))

        with wave.open(str(audio_file), "rb") as wf:
            duration_seconds = wf.getnframes() / wf.getframerate()
        if duration_seconds <= 0:
            raise AudioConversionError(f"Empty audio file: {selected_audio_file}")
        if duration_seconds <= 1200:
            return [audio_file]

        parts = _split_wav_20_minutes(audio_file, output_file)
        if not parts:
            raise OSError("ffmpeg did not create any audio segments")
        if not keep_wave:
            output_file.unlink(missing_ok=True)
        return parts
    except (AudioConversionError, OSError, wave.Error, EOFError) as error:
        try:
            output_file.unlink(missing_ok=True)
            for part in output_file.parent.glob(f"{output_file.stem}-part-*.wav"):
                part.unlink(missing_ok=True)
        except OSError as cleanup_error:
            error.add_note(f"Cannot remove temporary audio: {cleanup_error}")
        raise AudioConversionError(f"Cannot process audio {selected_audio_file}: {error}") from error
