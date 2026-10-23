"""CUDA detection and memory cleanup for the transcription runtime."""

import logging

import ctranslate2 as ct2
import torch


def is_cuda_available() -> bool:
    """Check CUDA through the inference backend, falling back to CPU on errors."""
    try:
        return ct2.get_cuda_device_count() > 0
    except Exception as error:
        logging.warning("CUDA device count check failed: %s", error)
        return False


def cleanup_cuda_memory() -> None:
    """Attempt to release CUDA caches without interrupting pipeline cleanup."""
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        logging.info("CUDA memory cleanup")
    except RuntimeError as error:
        logging.error("Failed to clean CUDA memory: %s", error)
