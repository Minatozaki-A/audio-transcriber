from unittest.mock import Mock

import numpy as np
import pytest

from errors import NoiseReductionError
from processor import cleaner


def test_noise_write_failure_preserves_cause_and_removes_partial(tmp_path, monkeypatch):
    source = tmp_path / "source.wav"
    source.write_bytes(b"original")
    output = tmp_path / "cleaned.wav"
    failure = OSError("disk full")
    monkeypatch.setattr(cleaner, "create_temp_audio_path", lambda directory: output)
    monkeypatch.setattr(cleaner.sf, "read", lambda *a, **k: (np.zeros(160), 16000))
    monkeypatch.setattr(cleaner.nr, "reduce_noise", lambda **kwargs: kwargs["y"])

    def write(*args):
        output.write_bytes(b"partial")
        raise failure

    monkeypatch.setattr(cleaner.sf, "write", write)
    with pytest.raises(NoiseReductionError) as caught:
        cleaner.reduce_noise(source, tmp_path)
    assert caught.value.__cause__ is failure
    assert not output.exists()
    assert source.read_bytes() == b"original"


def test_noise_read_failure_preserves_cause(tmp_path, monkeypatch):
    failure = OSError("cannot read")
    monkeypatch.setattr(cleaner.sf, "read", Mock(side_effect=failure))
    with pytest.raises(NoiseReductionError) as caught:
        cleaner.reduce_noise(tmp_path / "source.wav", tmp_path)
    assert caught.value.__cause__ is failure
