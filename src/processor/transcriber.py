import gc
import torch
import logging
import ctranslate2 as ct2
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from faster_whisper import WhisperModel


@contextmanager
def whisper_model(
        model_size: str,
        device: str,
        compute_type: str
    ) -> Iterator[WhisperModel]:
    """Load a WhisperModel and attempt memory cleanup on exit."""
    model = None

    try:
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
        logging.info("Model loaded successfully: %s (%s, %s)", model_size, device, compute_type)

        yield model

    finally:
        if model is not None:
            del model
            gc.collect()
            logging.info("Model garbage collected")

        if device == "cuda" and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                logging.info("CUDA memory cleanup")
            except RuntimeError as e:
                logging.error("Failed to clean CUDA memory: %s", e)


@contextmanager
def load_whisper_model() -> Iterator[WhisperModel]:
    try:
        cuda_available = ct2.get_cuda_device_count() > 0
    except Exception as e:
        logging.warning("CUDA device count check failed: %s", e)
        cuda_available = False

    if cuda_available:
        with ExitStack() as stack:
            try:
                logging.info("Cuda is available: loading model in DGPU with FP16")
                model = stack.enter_context(
                    whisper_model("medium", "cuda", compute_type="float16")
                )
            except (RuntimeError, OSError, ValueError) as e:
                logging.warning("Failed to load model in DGPU with FP16: %s", e)
            else:
                yield model
                return


    logging.info("Loading model in CPU with INT8")

    with whisper_model("small", "cpu", compute_type="int8") as model:
        yield model
