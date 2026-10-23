import subprocess
from unittest.mock import Mock

import pytest

from errors import AudioConversionError
from utils import helpers


def test_offsets_use_actual_durations(tmp_path, make_wav):
    paths = [make_wav(tmp_path / "a.wav", 2.25), make_wav(tmp_path / "b.wav", 1.5)]
    chunks = helpers.build_audio_chunks(paths)
    assert [chunk.path for chunk in chunks] == paths
    assert [chunk.offset_seconds for chunk in chunks] == [0, 2.25]


@pytest.mark.parametrize("seconds,split", [(900.5, False), (1199.5, False), (1200, False), (1200.5, True)])
def test_target_wav_conversion_boundary(tmp_path, make_wav, monkeypatch, seconds, split):
    source = make_wav(tmp_path / "source.wav", seconds)
    temporary = tmp_path / "temporary"
    temporary.mkdir()
    monkeypatch.setattr(helpers.magic, "from_file", lambda *a, **k: "audio/wav")
    part = temporary / "part.wav"
    splitter = Mock(return_value=[part])
    monkeypatch.setattr(helpers, "_split_wav_20_minutes", splitter)
    command = Mock()
    monkeypatch.setattr(helpers, "run_ffmpeg", command)
    assert helpers.convert_to_wav_16_mono(source, temporary) == ([part] if split else [source])
    assert splitter.call_count == int(split)
    command.assert_not_called()
    assert source.is_file()


def test_non_target_audio_is_converted(tmp_path, make_wav, monkeypatch):
    source = tmp_path / "source.mp3"
    source.write_bytes(b"original")
    monkeypatch.setattr(helpers.magic, "from_file", lambda *a, **k: "audio/mpeg")

    def convert(command):
        assert command[command.index("-i") + 1] == str(source)
        make_wav(helpers.Path(command[-1]))

    monkeypatch.setattr(helpers, "run_ffmpeg", convert)
    paths = helpers.convert_to_wav_16_mono(source, tmp_path)
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
    monkeypatch.setattr(helpers.magic, "from_file", lambda *a, **k: "audio/mpeg")
    monkeypatch.setattr(helpers, "create_temp_audio_path", lambda directory: converted)
    monkeypatch.setattr(helpers, "run_ffmpeg", lambda command: make_wav(converted, 1201))
    parts = [temporary / "part-000.wav", temporary / "part-001.wav"]

    def split(audio, output):
        assert audio == converted
        return [make_wav(part) for part in parts]

    monkeypatch.setattr(helpers, "_split_wav_20_minutes", split)
    assert helpers.convert_to_wav_16_mono(source, temporary, keep_wave=keep_wave) == parts
    assert converted.exists() == keep_wave
    assert all(part.is_file() for part in parts)
    assert source.read_bytes() == b"original"


def test_split_error_removes_generated_parts_preserves_source(tmp_path, make_wav, monkeypatch):
    source = make_wav(tmp_path / "source.wav", 1201)
    temporary = tmp_path / "temporary"
    temporary.mkdir()
    monkeypatch.setattr(helpers.magic, "from_file", lambda *a, **k: "audio/wav")
    failure = AudioConversionError("split failed")

    def split(audio, output):
        output.with_name(f"{output.stem}-part-000.wav").write_bytes(b"partial")
        raise failure

    monkeypatch.setattr(helpers, "_split_wav_20_minutes", split)
    with pytest.raises(AudioConversionError, match="split failed") as caught:
        helpers.convert_to_wav_16_mono(source, temporary)
    assert caught.value.__cause__ is failure
    assert list(temporary.iterdir()) == []
    assert source.is_file()


@pytest.mark.parametrize("failure,message", [
    (FileNotFoundError("missing"), "ffmpeg not found"),
    (subprocess.CalledProcessError(1, ["ffmpeg"], stderr="invalid stream"), "invalid stream"),
])
def test_ffmpeg_errors_preserve_cause(monkeypatch, failure, message):
    monkeypatch.setattr(helpers.sp, "run", Mock(side_effect=failure))
    with pytest.raises(AudioConversionError, match=message) as caught:
        helpers.run_ffmpeg(["ffmpeg"])
    assert caught.value.__cause__ is failure


def test_unsupported_mime_is_error(tmp_path, monkeypatch):
    monkeypatch.setattr(helpers.magic, "from_file", lambda *a, **k: "text/plain")
    with pytest.raises(AudioConversionError, match="Unsupported file type"):
        helpers.convert_to_wav_16_mono(tmp_path / "text.txt", tmp_path)


def test_publish_preserves_existing_transcript(tmp_path):
    output = tmp_path / "transcript.md"
    output.write_text("existing", encoding="utf-8")
    draft = tmp_path / "draft.md"
    draft.write_text("voz con acentos: canción", encoding="utf-8")
    published = helpers.publish_transcript(draft, output)
    assert published.name == "transcript-(1).md"
    assert published.read_text(encoding="utf-8") == draft.read_text(encoding="utf-8")
    assert output.read_text(encoding="utf-8") == "existing"


def test_publish_retries_concurrent_collision(tmp_path, monkeypatch):
    output = tmp_path / "transcript.md"
    draft = tmp_path / "draft.md"
    draft.write_text("new", encoding="utf-8")
    unique_path = helpers._unique_path
    calls = 0

    def racing_path(path):
        nonlocal calls
        candidate = unique_path(path)
        if calls == 0:
            candidate.write_text("concurrent", encoding="utf-8")
        calls += 1
        return candidate

    monkeypatch.setattr(helpers, "_unique_path", racing_path)
    published = helpers.publish_transcript(draft, output)
    assert published.read_text(encoding="utf-8") == "new"
    assert output.read_text(encoding="utf-8") == "concurrent"
    assert calls == 2


def test_publish_failure_removes_partial_file(tmp_path, monkeypatch):
    output = tmp_path / "transcript.md"
    draft = tmp_path / "draft.md"
    draft.write_text("new", encoding="utf-8")
    failure = OSError("disk full")

    def fail(source, destination):
        destination.write("partial")
        raise failure

    monkeypatch.setattr(helpers.shutil, "copyfileobj", fail)
    with pytest.raises(OSError) as caught:
        helpers.publish_transcript(draft, output)
    assert caught.value is failure
    assert not output.exists()


def test_conversion_failure_removes_partial_wav(tmp_path, monkeypatch):
    source = tmp_path / "source.mp3"
    source.write_bytes(b"original")
    temporary = tmp_path / "temporary"
    temporary.mkdir()
    failure = AudioConversionError("invalid stream")
    monkeypatch.setattr(helpers.magic, "from_file", lambda *a, **k: "audio/mpeg")

    def convert(command):
        helpers.Path(command[-1]).write_bytes(b"partial")
        raise failure

    monkeypatch.setattr(helpers, "run_ffmpeg", convert)
    with pytest.raises(AudioConversionError) as caught:
        helpers.convert_to_wav_16_mono(source, temporary)
    assert caught.value.__cause__ is failure
    assert list(temporary.iterdir()) == []
    assert source.read_bytes() == b"original"


def test_mime_detection_failure_preserves_cause(tmp_path, monkeypatch):
    failure = OSError("permission denied")
    monkeypatch.setattr(helpers.magic, "from_file", Mock(side_effect=failure))
    with pytest.raises(AudioConversionError) as caught:
        helpers.convert_to_wav_16_mono(tmp_path / "source.wav", tmp_path)
    assert caught.value.__cause__ is failure


def test_empty_audio_is_rejected(tmp_path, make_wav, monkeypatch):
    source = make_wav(tmp_path / "empty.wav", 0)
    monkeypatch.setattr(helpers.magic, "from_file", lambda *a, **k: "audio/wav")
    with pytest.raises(AudioConversionError, match="Empty audio"):
        helpers.convert_to_wav_16_mono(source, tmp_path)
    assert source.is_file()


def test_split_uses_twenty_minutes(tmp_path, monkeypatch):
    output = tmp_path / "converted.wav"
    part = tmp_path / "converted-part-000.wav"
    part.touch()
    command = Mock()
    monkeypatch.setattr(helpers, "run_ffmpeg", command)
    assert helpers._split_wav_20_minutes(tmp_path / "source.wav", output) == [part]
    arguments = command.call_args.args[0]
    assert arguments[arguments.index("-segment_time") + 1] == "1200"
