from unittest.mock import Mock

import pytest

from models import runtime


@pytest.mark.parametrize("operation", ["empty_cache", "ipc_collect"])
def test_cuda_cleanup_failure_is_logged(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, operation: str,
) -> None:
    monkeypatch.setattr(runtime.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(runtime.torch.cuda, "empty_cache", Mock())
    monkeypatch.setattr(runtime.torch.cuda, "ipc_collect", Mock())
    monkeypatch.setattr(
        runtime.torch.cuda, operation, Mock(side_effect=RuntimeError("CUDA cleanup failed"))
    )

    runtime.cleanup_cuda_memory()

    assert "Failed to clean CUDA memory: CUDA cleanup failed" in caplog.text
