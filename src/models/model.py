"""Whisper model lifetime and GPU-to-CPU loading fallback."""

import gc
import logging
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager

from faster_whisper import WhisperModel

from models.runtime import cleanup_cuda_memory, is_cuda_available


@contextmanager
def whisper_model(
    model_size: str, device: str, compute_type: str,
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
        if device == "cuda":
            cleanup_cuda_memory()


@contextmanager
def load_whisper_model() -> Iterator[WhisperModel]:
    """Prefer medium CUDA/FP16, using small CPU/INT8 if loading is unavailable."""
    if is_cuda_available():
        with ExitStack() as stack:
            try:
                logging.info("Cuda is available: loading model in DGPU with FP16")
                model = stack.enter_context(
                    whisper_model("medium", "cuda", compute_type="float16")
                )
            except (RuntimeError, OSError, ValueError) as error:
                logging.warning("Failed to load model in DGPU with FP16: %s", error)
            else:
                yield model
                return

    logging.info("Loading model in CPU with INT8")
    with whisper_model("small", "cpu", compute_type="int8") as model:
        yield model
