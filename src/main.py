import logging
import sys
from pathlib import Path
from utils.helpers import convert_to_wav_16_mono, remove_temp_files
from processor.cleaner import reduce_noise, audio_normalize
from processor.transcriber import transcribe_audio


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def main() -> None:

    path_audio: Path | None = convert_to_wav_16_mono()
    if not path_audio:
        logging.error("No audio file found or conversion failed.")
        sys.exit(1)

    path_audio_nr: Path | None = reduce_noise(path_audio)
    if not path_audio_nr:
        logging.error("Noise reduction failed.")
        sys.exit(1)

    """    
    path_audio_normalize: Path | None = audio_normalize(path_audio_nr)
    if not path_audio_normalize:
        logging.error("Audio normalization failed.")
        sys.exit(1)
    """

    transcribe_audio(path_audio_nr)

    remove_temp_files()

if __name__ == "__main__":
    main()
