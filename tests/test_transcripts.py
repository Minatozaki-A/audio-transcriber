import pytest

from utils import transcripts


def test_publish_preserves_existing_transcript(tmp_path):
    output = tmp_path / "transcript.md"
    output.write_text("existing", encoding="utf-8")
    draft = tmp_path / "draft.md"
    draft.write_text("voz con acentos: canción", encoding="utf-8")
    published = transcripts.publish_transcript(draft, output)
    assert published.name == "transcript-(1).md"
    assert published.read_text(encoding="utf-8") == draft.read_text(encoding="utf-8")
    assert output.read_text(encoding="utf-8") == "existing"


def test_publish_retries_concurrent_collision(tmp_path, monkeypatch):
    output = tmp_path / "transcript.md"
    draft = tmp_path / "draft.md"
    draft.write_text("new", encoding="utf-8")
    unique_path = transcripts._unique_path
    calls = 0

    def racing_path(path):
        nonlocal calls
        candidate = unique_path(path)
        if calls == 0:
            candidate.write_text("concurrent", encoding="utf-8")
        calls += 1
        return candidate

    monkeypatch.setattr(transcripts, "_unique_path", racing_path)
    published = transcripts.publish_transcript(draft, output)
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

    monkeypatch.setattr(transcripts.shutil, "copyfileobj", fail)
    with pytest.raises(OSError) as caught:
        transcripts.publish_transcript(draft, output)
    assert caught.value is failure
    assert not output.exists()
