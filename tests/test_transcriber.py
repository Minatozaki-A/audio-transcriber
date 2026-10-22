from unittest.mock import Mock

import pytest

from processor import transcriber


def test_load_whisper_model_uses_cpu_without_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    model = object()
    create_model = Mock(return_value=model)
    empty_cache = Mock()
    ipc_collect = Mock()
    monkeypatch.setattr(transcriber.ct2, "get_cuda_device_count", lambda: 0)
    monkeypatch.setattr(transcriber, "WhisperModel", create_model)
    monkeypatch.setattr(transcriber.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(transcriber.torch.cuda, "empty_cache", empty_cache)
    monkeypatch.setattr(transcriber.torch.cuda, "ipc_collect", ipc_collect)

    with transcriber.load_whisper_model() as loaded:
        assert loaded is model

    create_model.assert_called_once_with("small", device="cpu", compute_type="int8")
    empty_cache.assert_not_called()
    ipc_collect.assert_not_called()


def test_load_whisper_model_uses_gpu_and_cleans_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    model = object()
    create_model = Mock(return_value=model)
    empty_cache = Mock()
    ipc_collect = Mock()
    monkeypatch.setattr(transcriber.ct2, "get_cuda_device_count", lambda: 1)
    monkeypatch.setattr(transcriber, "WhisperModel", create_model)
    monkeypatch.setattr(transcriber.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(transcriber.torch.cuda, "empty_cache", empty_cache)
    monkeypatch.setattr(transcriber.torch.cuda, "ipc_collect", ipc_collect)

    with transcriber.load_whisper_model() as loaded:
        assert loaded is model

    create_model.assert_called_once_with("medium", device="cuda", compute_type="float16")
    empty_cache.assert_called_once_with()
    ipc_collect.assert_called_once_with()


@pytest.mark.parametrize("gpu_error", [RuntimeError("CUDA failed"), ValueError("unsupported FP16")])
def test_gpu_load_failure_falls_back_to_cpu(
    monkeypatch: pytest.MonkeyPatch, gpu_error: Exception
) -> None:
    cpu_model = object()

    def create_model(size: str, *, device: str, compute_type: str) -> object:
        if device == "cuda":
            raise gpu_error
        assert (size, device, compute_type) == ("small", "cpu", "int8")
        return cpu_model

    create = Mock(side_effect=create_model)
    monkeypatch.setattr(transcriber.ct2, "get_cuda_device_count", lambda: 1)
    monkeypatch.setattr(transcriber, "WhisperModel", create)
    monkeypatch.setattr(transcriber.torch.cuda, "is_available", lambda: False)

    with transcriber.load_whisper_model() as loaded:
        assert loaded is cpu_model

    assert create.call_count == 2


def test_cuda_detection_failure_uses_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    def detection_error() -> int:
        raise RuntimeError("CUDA driver unavailable")

    create_model = Mock(return_value=object())
    monkeypatch.setattr(transcriber.ct2, "get_cuda_device_count", detection_error)
    monkeypatch.setattr(transcriber, "WhisperModel", create_model)

    with transcriber.load_whisper_model():
        pass

    create_model.assert_called_once_with("small", device="cpu", compute_type="int8")


def test_gpu_usage_error_does_not_start_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    create_model = Mock(return_value=object())
    monkeypatch.setattr(transcriber.ct2, "get_cuda_device_count", lambda: 1)
    monkeypatch.setattr(transcriber, "WhisperModel", create_model)
    monkeypatch.setattr(transcriber.torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="transcription failed"):
        with transcriber.load_whisper_model():
            raise RuntimeError("transcription failed")

    create_model.assert_called_once_with("medium", device="cuda", compute_type="float16")
