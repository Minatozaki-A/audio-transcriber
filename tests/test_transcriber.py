from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from processor import transcriber
from processor.chunks import AudioChunk


@pytest.mark.parametrize("keep_wave", [False, True])
@pytest.mark.parametrize("temporary_input", [False, True])
def test_closing_transcript_lines_respects_audio_ownership(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    keep_wave: bool, temporary_input: bool,
) -> None:
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    source = (audio_dir if temporary_input else tmp_path) / "source.wav"
    source.write_bytes(b"source")
    cleaned = audio_dir / "cleaned.wav"
    cleaned.write_bytes(b"cleaned")
    monkeypatch.setattr(transcriber, "reduce_noise", lambda *args: cleaned)
    model = Mock()
    model.transcribe.return_value = (
        iter([
            SimpleNamespace(start=0.5, end=1.0, text=" canción"),
            SimpleNamespace(start=1.0, end=2.0, text=" sin consumir"),
        ]),
        None,
    )

    lines = transcriber.transcript_lines(
        model, [AudioChunk(source, 300.0)], audio_dir, None,
        noise_reduce=True, keep_wave=keep_wave,
    )
    try:
        assert next(lines) == "[300.50s -> 301.00s]  canción\n"
        assert source.is_file()
        assert cleaned.is_file()
    finally:
        lines.close()

    assert source.exists() == (keep_wave or not temporary_input)
    assert cleaned.exists() == keep_wave
    model.transcribe.assert_called_once_with(str(cleaned), beam_size=5, language=None)
