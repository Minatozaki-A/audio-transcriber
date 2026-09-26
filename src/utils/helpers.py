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



def convert_to_wav_16_mono() -> list[Path]:
    """Prompt the user to pick files, convert each to 16 kHz mono WAV, and return the resulting paths."""
    audio_files_converted: list[Path] = []
    # audio_files: list[Path] = select_audio_files()
    audio_files: list[Path] = []


    for af in audio_files:
        try:
            mime: str = magic.from_file(af, mime=True)
        except OSError:
            logging.error("Cannot read file %s", af.name)
            continue


        except magic.MagicException:
            logging.exception("Cannot detect MIME type for file %s", af)
            continue

        if not (mime.startswith("audio/") or mime.startswith("video/")):
            logging.error("Unsupported file type: %s (%s)", mime, af.name)
            continue


        if mime in {"audio/wav", "audio/x-wav"}:
            if _is_target_wav_format_stdlib(af):
                logging.info("File %s already in target WAV format, skipping conversion", af.name)
                audio_files_converted.append(af)
                continue
            logging.info("File %s is WAV but not target format, converting", af.name)


        new_name: Path = create_temp_audio_path()

        try:
            sp.run(_command_ffmpeg(af, new_name), check=True)
            logging.info("%s converted into %s", af.name, new_name.name)

        except sp.CalledProcessError as e:
            logging.error("Error converting audio file %s: %s", af.name, e)
            continue

        except FileNotFoundError:
            logging.error("ffmpeg not found in PATH — aborting conversion")
            raise

        except OSError as e:
            logging.error("I/O error converting %s: %s", af.name, e)
            continue

        logging.info("Audio file %s | type %s | converted to %s", af, mime, new_name.name)
        audio_files_converted.append(new_name)

    return audio_files_converted
