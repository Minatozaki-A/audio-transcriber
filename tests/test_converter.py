import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest

from errors import AudioConversionError
from processor import converter


@pytest.mark.parametrize("seconds,split", [(900.5, False), (1199.5, False), (1200, False), (1200.5, True)])
def test_target_wav_conversion_boundary(tmp_path, make_wav, monkeypatch, seconds, split):
    source = make_wav(tmp_path / "source.wav", seconds)
    temporary = tmp_path / "temporary"
    temporary.mkdir()
    monkeypatch.setattr(converter.magic, "from_file", lambda *a, **k: "audio/wav")
    part = temporary / "part.wav"
    splitter = Mock(return_value=[part])
    monkeypatch.setattr(converter, "_split_wav_20_minutes", splitter)
    command = Mock()
    monkeypatch.setattr(converter, "run_ffmpeg", command)
    assert converter.convert_to_wav_16_mono(source, temporary) == ([part] if split else [source])
    assert splitter.call_count == int(split)
    command.assert_not_called()
    assert source.is_file()


def test_non_target_audio_is_converted(tmp_path, make_wav, monkeypatch):
    source = tmp_path / "source.mp3"
    source.write_bytes(b"original")
    monkeypatch.setattr(converter.magic, "from_file", lambda *a, **k: "audio/mpeg")

    def convert(command):
        assert command[command.index("-i") + 1] == str(source)
        make_wav(Path(command[-1]))

    monkeypatch.setattr(converter, "run_ffmpeg", convert)
    paths = converter.convert_to_wav_16_mono(source, tmp_path)
    assert len(paths) == 1
    assert paths[0] != source
    assert source.read_bytes() == b"original"


@pytest.mark.parametrize("keep_wave", [False, True])
def test_converted_wav_retention_after_splitting(tmp_path, make_wav, monkeypatch, keep_wave):
    source = tmp_path / "source.mp3"
    source.write_bytes(b"original")
    temporary = tmp_path / "temporary"
    temporary.mkdir()
    converted = temporary / "converted.wav"
    monkeypatch.setattr(converter.magic, "from_file", lambda *a, **k: "audio/mpeg")
    monkeypatch.setattr(converter, "create_temp_audio_path", lambda directory: converted)
    monkeypatch.setattr(converter, "run_ffmpeg", lambda command: make_wav(converted, 1201))
    parts = [temporary / "part-000.wav", temporary / "part-001.wav"]

    def split(audio, output):
        assert audio == converted
        return [make_wav(part) for part in parts]

    monkeypatch.setattr(converter, "_split_wav_20_minutes", split)
    assert converter.convert_to_wav_16_mono(source, temporary, keep_wave=keep_wave) == parts
    assert converted.exists() == keep_wave
    assert all(part.is_file() for part in parts)
    assert source.read_bytes() == b"original"


def test_split_error_removes_generated_parts_preserves_source(tmp_path, make_wav, monkeypatch):
    source = make_wav(tmp_path / "source.wav", 1201)
    temporary = tmp_path / "temporary"
    temporary.mkdir()
    monkeypatch.setattr(converter.magic, "from_file", lambda *a, **k: "audio/wav")
    failure = AudioConversionError("split failed")

    def split(audio, output):
        output.with_name(f"{output.stem}-part-000.wav").write_bytes(b"partial")
        raise failure

    monkeypatch.setattr(converter, "_split_wav_20_minutes", split)
    with pytest.raises(AudioConversionError, match="split failed") as caught:
        converter.convert_to_wav_16_mono(source, temporary)
    assert caught.value.__cause__ is failure
    assert list(temporary.iterdir()) == []
    assert source.is_file()


@pytest.mark.parametrize("failure,message", [
    (FileNotFoundError("missing"), "ffmpeg not found"),
    (subprocess.CalledProcessError(1, ["ffmpeg"], stderr="invalid stream"), "invalid stream"),
])
def test_ffmpeg_errors_preserve_cause(monkeypatch, failure, message):
    monkeypatch.setattr(converter.sp, "run", Mock(side_effect=failure))
    with pytest.raises(AudioConversionError, match=message) as caught:
        converter.run_ffmpeg(["ffmpeg"])
    assert caught.value.__cause__ is failure


def test_unsupported_mime_is_error(tmp_path, monkeypatch):
    monkeypatch.setattr(converter.magic, "from_file", lambda *a, **k: "text/plain")
    with pytest.raises(AudioConversionError, match="Unsupported file type"):
        converter.convert_to_wav_16_mono(tmp_path / "text.txt", tmp_path)


def test_conversion_failure_removes_partial_wav(tmp_path, monkeypatch):
    source = tmp_path / "source.mp3"
    source.write_bytes(b"original")
    temporary = tmp_path / "temporary"
    temporary.mkdir()
    failure = AudioConversionError("invalid stream")
    monkeypatch.setattr(converter.magic, "from_file", lambda *a, **k: "audio/mpeg")

    def convert(command):
        Path(command[-1]).write_bytes(b"partial")
        raise failure

    monkeypatch.setattr(converter, "run_ffmpeg", convert)
    with pytest.raises(AudioConversionError) as caught:
        converter.convert_to_wav_16_mono(source, temporary)
    assert caught.value.__cause__ is failure
    assert list(temporary.iterdir()) == []
    assert source.read_bytes() == b"original"


def test_mime_detection_failure_preserves_cause(tmp_path, monkeypatch):
    failure = OSError("permission denied")
    monkeypatch.setattr(converter.magic, "from_file", Mock(side_effect=failure))
    with pytest.raises(AudioConversionError) as caught:
        converter.convert_to_wav_16_mono(tmp_path / "source.wav", tmp_path)
    assert caught.value.__cause__ is failure


def test_empty_audio_is_rejected(tmp_path, make_wav, monkeypatch):
    source = make_wav(tmp_path / "empty.wav", 0)
    monkeypatch.setattr(converter.magic, "from_file", lambda *a, **k: "audio/wav")
    with pytest.raises(AudioConversionError, match="Empty audio"):
        converter.convert_to_wav_16_mono(source, tmp_path)
    assert source.is_file()


def test_split_uses_twenty_minutes(tmp_path, monkeypatch):
    output = tmp_path / "converted.wav"
    part = tmp_path / "converted-part-000.wav"
    part.touch()
    command = Mock()
    monkeypatch.setattr(converter, "run_ffmpeg", command)
    assert converter._split_wav_20_minutes(tmp_path / "source.wav", output) == [part]
    arguments = command.call_args.args[0]
    assert arguments[arguments.index("-segment_time") + 1] == "1200"
