import gc
import torch
import logging
from faster_whisper import WhisperModel
from pathlib import Path

_MODEL_SIZE: str = "small" # "medium" or "large-v3"

def transcribe_audio(path_audio: Path) -> None:

    path_audio_nr = path_audio

    try:
        # Run on GPU with FP16
        # model = WhisperModel(_MODEL_SIZE, device="cuda", compute_type="float16")

        # or run on GPU with INT8
        # model = WhisperModel(_MODEL_SIZE, device="cuda", compute_type="int8_float16")

        # or run on CPU with INT8
        model = WhisperModel(_MODEL_SIZE, device="cpu", compute_type="int8")

        segments, info = model.transcribe(str(path_audio_nr), beam_size=5)

        logging.info("Detected language '%s' with probability %f" % (info.language, info.language_probability))

        for segment in segments:
            print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))

    except Exception as e:
        logging.error(f"Error transcribing audio: {e}")

    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()