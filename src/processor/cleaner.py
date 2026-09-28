import noisereduce as nr
import soundfile as sf
# import librosa
import logging
from pathlib import Path
from utils.helpers import create_temp_audio_path


def reduce_noise(path_file: Path, tempd:str) -> Path | None:
    output_path: Path = create_temp_audio_path(tempd)

    try:
        data, rate = sf.read(path_file, dtype="float32")
    except FileNotFoundError as e:
        logging.error("File %s not found: %s", path_file, e)
        return None
    except sf.SoundFileError as e:
        logging.error("Error reading file %s: %s", path_file, e)
        return None


    try:
        if data.ndim >= 2:
            data = data.mean(axis=1)
        reduced_noise = nr.reduce_noise(y=data, sr=rate, thresh_n_mult_nonstationary=2, stationary=False)
    except ValueError as e:
        logging.error("Error processing audio %s: %s", path_file, e)
        return None
    except MemoryError as e:
        logging.error("Out of memory processing %s: %s", path_file, e)
        return None

    try:
        sf.write(str(output_path), reduced_noise, rate)
    except (sf.SoundFileError, OSError) as e:
        logging.error("Error writing file %s: %s", output_path, e)
        try:
            output_path.unlink(missing_ok=True)
        except OSError as cleanup_error:
            logging.error("Cannot remove partial file %s: %s", output_path, cleanup_error)
        return None

    return output_path



"""
def audio_normalize(path_file: Path) -> Path | None:

    data, rate = librosa.load(path_file, sr=None)
    audio = librosa.util.normalize(data, norm=1)
    output_path: Path = generate_name_audio_file()

    try:
        sf.write(str(output_path), audio, rate)
    except Exception as e:
        logging.error(f"Error writing file %s: %s", output_path, e)
        return None
"""