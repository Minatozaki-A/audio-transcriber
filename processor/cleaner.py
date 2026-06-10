import noisereduce as nr
import soundfile as sf
import librosa
import logging
from pathlib import Path
from utils.helpers import generate_name_audio_file


_TEMP_DIR: Path = Path(__file__).parent.parent / "temp"


def reduce_noise(path_file: Path) -> Path | None:
    data, rate = sf.read(path_file)

    reduced_noise = nr.reduce_noise(y=data, sr=rate, thresh_n_mult_nonstationary=2, stationary=False)

    output_path: Path = generate_name_audio_file(_TEMP_DIR)

    try:
        sf.write(str(output_path), reduced_noise, rate)
    except Exception as e:
        logging.error(f"Error writing file {output_path}: {e}")
    return output_path


def coso_librosa(path_file: Path) -> None:
    data, rate = librosa.load(path_file)



