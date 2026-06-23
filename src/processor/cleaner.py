import noisereduce as nr
import soundfile as sf
import librosa
import logging
import tempfile
from pathlib import Path
from utils.helpers import generate_name_audio_file



_TEMP_DIR: Path = Path(tempfile.gettempdir())


def reduce_noise(path_file: Path) -> Path | None:
    output_path: Path = generate_name_audio_file()

    try:
        data, rate = sf.read(path_file)

        reduced_noise = nr.reduce_noise(y=data, sr=rate, thresh_n_mult_nonstationary=2, stationary=False)

        sf.write(str(output_path), reduced_noise, rate)

        return output_path

    except FileNotFoundError as e:
        logging.error("File s% not found: %s", path_file, e)

    except sf.SoundFileError as e:
        logging.error("Error reading file %s: %s", path_file, e)

    except Exception as e:
        logging.error("Error writing file %s: %s", output_path, e)

    return None



def audio_normalize(path_file: Path) -> Path | None:
    data, rate = librosa.load(path_file, sr=None)
    audio = librosa.util.normalize(data, norm=1)
    output_path: Path = generate_name_audio_file()

    try:
        sf.write(str(output_path), audio, rate)
    except Exception as e:
        logging.error(f"Error writing file {output_path}: {e}")

    return output_path