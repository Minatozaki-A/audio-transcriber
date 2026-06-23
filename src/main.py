import logging
import sys
from pathlib import Path
from utils.helpers import convert_to_wav_16_mono_v2
from processor.cleaner import reduce_noise
from processor.transcriber import whisper_model


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def main() -> None:

    audio_files: list[Path] = convert_to_wav_16_mono_v2()
    if not audio_files:
        logging.error("No audio files found or conversion failed.")
        sys.exit(1)

    audio_files_nr: list[Path] = []
    for af in audio_files:
        path_audio_nr: Path | None = reduce_noise(af)
        if not path_audio_nr:
            logging.warning("Noise reduction failed for %s, skipping.", af.name)
            continue
        audio_files_nr.append(path_audio_nr)

    if not audio_files_nr:
        logging.error("All noise reduction steps failed.")
        sys.exit(1)

    # Run on GPU with FP16
    # whisper_model("medium", device="cuda", compute_type="float16")

    # or run on GPU with INT8
    # whisper_model("medium", device="cuda", compute_type="int8_float16")

    # or run on CPU with INT8
    # whisper_model("small", device="cpu", compute_type="int8")

    with whisper_model("small",  device="cpu", compute_type="int8") as model:
        segments, _ = model.transcribe(str(path_audio_nr), beam_size=5)
        # result = " ".join(seg.text for seg in segments)
        for segment in segments:
            print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))


if __name__ == "__main__":
    main()
