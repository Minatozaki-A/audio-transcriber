from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import main as pipeline
from errors import NoiseReductionError, TranscriptionError


def test_pipeline_accumulates_timestamps_and_preserves_text(tmp_path, make_wav, monkeypatch):
    source = make_wav(tmp_path / "source.wav")
    temporary_paths = []

    def convert(path, directory):
        temporary_paths.append(directory)
        return [make_wav(directory / "a.wav", 2.25), make_wav(directory / "b.wav", 1)]

    monkeypatch.setattr(pipeline, "convert_to_wav_16_mono", convert)
    monkeypatch.setattr(pipeline, "reduce_noise", lambda path, directory: path)
    model = Mock()
    model.supported_languages = ["es", "en", "ja"]
    model.transcribe.side_effect = [
        (iter([SimpleNamespace(start=0.5, end=1.5, text=" primera frase")]), None),
        (iter([SimpleNamespace(start=0.25, end=0.75, text=" canción")]), None),
    ]
    closed = []

    @contextmanager
    def load_model():
        try:
            yield model
        finally:
            closed.append(True)

    monkeypatch.setattr(pipeline, "load_whisper_model", load_model)
    output = pipeline.transcribe_file(source, tmp_path / "result.md", language="es")
    assert output.read_text(encoding="utf-8") == (
        "[0.50s -> 1.50s]  primera frase\n[2.50s -> 3.00s]  canción\n"
    )
    assert all(call.kwargs["language"] == "es" for call in model.transcribe.call_args_list)
    assert closed == [True]
    assert source.is_file()
    assert all(not directory.exists() for directory in temporary_paths)


@pytest.mark.parametrize("stage", ["load", "call", "iterate", "clean"])
def test_pipeline_failures_preserve_cause_and_cleanup(tmp_path, make_wav, monkeypatch, stage):
    source = make_wav(tmp_path / "source.wav")
    output = tmp_path / "result.md"
    output.write_text("existing", encoding="utf-8")
    temporary_paths = []
    failure = NoiseReductionError("noise failed") if stage == "clean" else RuntimeError("model failed")

    def convert(path, directory):
        temporary_paths.append(directory)
        return [make_wav(directory / "part.wav")]

    def clean(path, directory):
        if stage == "clean":
            raise failure
        return path

    def segments():
        yield SimpleNamespace(start=0, end=0.5, text="partial")
        raise failure

    model = Mock()
    model.supported_languages = ["es", "en", "ja"]
    model.transcribe = Mock(side_effect=failure) if stage == "call" else Mock(return_value=(segments(), None))
    closed = []

    @contextmanager
    def load_model():
        if stage == "load":
            raise failure
        try:
            yield model
        finally:
            closed.append(True)

    monkeypatch.setattr(pipeline, "convert_to_wav_16_mono", convert)
    monkeypatch.setattr(pipeline, "reduce_noise", clean)
    monkeypatch.setattr(pipeline, "load_whisper_model", load_model)
    expected = NoiseReductionError if stage == "clean" else TranscriptionError
    with pytest.raises(expected) as caught:
        pipeline.transcribe_file(source, output, noise_reduce=True)
    assert (caught.value is failure) if stage == "clean" else (caught.value.__cause__ is failure)
    assert source.is_file()
    assert output.read_text(encoding="utf-8") == "existing"
    assert not (tmp_path / "result-(1).md").exists()
    assert all(not directory.exists() for directory in temporary_paths)
    assert closed == ([] if stage == "load" else [True])


def test_cli_returns_failure(tmp_path, make_wav, monkeypatch):
    source = make_wav(tmp_path / "source.wav")
    monkeypatch.setattr(pipeline, "create_name_trans_path", lambda name: tmp_path / "result.md")
    monkeypatch.setattr(pipeline, "transcribe_file", Mock(side_effect=TranscriptionError("failed")))
    assert pipeline.main([str(source), "result"]) == 1


@pytest.mark.parametrize("options,language", [([], None), (["--language", "es"], "es")])
def test_cli_passes_language_to_pipeline(tmp_path, make_wav, monkeypatch, options, language):
    source = make_wav(tmp_path / "source.wav")
    output = tmp_path / "result.md"
    transcribe = Mock(return_value=output)
    monkeypatch.setattr(pipeline, "create_name_trans_path", lambda name: output)
    monkeypatch.setattr(pipeline, "transcribe_file", transcribe)

    assert pipeline.main([str(source), "result", *options]) == 0
    transcribe.assert_called_once_with(source, output, language=language, noise_reduce=False)


def test_draft_write_failure_closes_model_and_keeps_io_error(tmp_path, make_wav, monkeypatch):
    from io import StringIO
    from pathlib import Path

    source = make_wav(tmp_path / "source.wav")
    failure = OSError("disk full")
    closed = []
    original_open = Path.open

    class FailingDraft(StringIO):
        def write(self, text):
            raise failure

    def open_file(path, *args, **kwargs):
        if path.name == "transcript.md" and args and args[0] == "w":
            return FailingDraft()
        return original_open(path, *args, **kwargs)

    model = Mock()
    model.supported_languages = ["es", "en", "ja"]
    model.transcribe.return_value = (
        iter([SimpleNamespace(start=0, end=1, text="text")]), None
    )

    @contextmanager
    def load_model():
        try:
            yield model
        finally:
            closed.append(True)

    monkeypatch.setattr(pipeline, "convert_to_wav_16_mono", lambda *args: [source])
    monkeypatch.setattr(pipeline, "reduce_noise", lambda path, directory: path)
    monkeypatch.setattr(pipeline, "load_whisper_model", load_model)
    monkeypatch.setattr(Path, "open", open_file)
    with pytest.raises(OSError) as caught:
        pipeline.transcribe_file(source, tmp_path / "output.md")
    assert caught.value is failure
    assert closed == [True]
    assert source.is_file()
    assert not (tmp_path / "output.md").exists()


@pytest.mark.parametrize("noise_reduce", [False, True])
def test_processed_chunks_are_removed_before_next_chunk(
    tmp_path, make_wav, monkeypatch, noise_reduce
):
    source = make_wav(tmp_path / "source.wav")
    paths = []
    cleaned = []

    def convert(path, directory):
        paths.extend([make_wav(directory / "a.wav"), make_wav(directory / "b.wav")])
        return paths

    def clean(path, directory):
        result = make_wav(directory / f"clean-{path.name}")
        cleaned.append(result)
        return result

    cleaner = Mock(side_effect=clean)
    calls = 0

    def transcribe(path, **kwargs):
        nonlocal calls
        if calls:
            assert not paths[0].exists()
            if noise_reduce:
                assert not cleaned[0].exists()
        expected = cleaned[calls] if noise_reduce else paths[calls]
        assert path == str(expected)
        assert expected.is_file()
        calls += 1

        def segments():
            assert expected.is_file()
            yield SimpleNamespace(start=0, end=1, text="text")
            assert expected.is_file()

        return segments(), None

    @contextmanager
    def load_model():
        yield SimpleNamespace(transcribe=transcribe)
        assert all(not path.exists() for path in paths + cleaned)

    monkeypatch.setattr(pipeline, "convert_to_wav_16_mono", convert)
    monkeypatch.setattr(pipeline, "reduce_noise", cleaner)
    monkeypatch.setattr(pipeline, "load_whisper_model", load_model)
    pipeline.transcribe_file(source, tmp_path / "out.md", noise_reduce=noise_reduce)
    assert cleaner.call_count == (2 if noise_reduce else 0)
    assert source.is_file()


def test_cli_enables_noise_reduction(tmp_path, make_wav, monkeypatch):
    source = make_wav(tmp_path / "source.wav")
    output = tmp_path / "out.md"
    transcribe = Mock(return_value=output)
    monkeypatch.setattr(pipeline, "create_name_trans_path", lambda name: output)
    monkeypatch.setattr(pipeline, "transcribe_file", transcribe)
    assert pipeline.main([str(source), "out", "--noise-reduce"]) == 0
    transcribe.assert_called_once_with(source, output, language=None, noise_reduce=True)


@pytest.mark.parametrize("language,supported", [("jap", ["ja", "en"]), ("es", ["en"])])
def test_invalid_language_stops_before_audio_processing(
    tmp_path, make_wav, monkeypatch, caplog, language, supported
):
    source = make_wav(tmp_path / "source.wav")
    output = tmp_path / "out.md"
    model = Mock(supported_languages=supported)
    closed = []

    @contextmanager
    def load_model():
        try:
            yield model
        finally:
            closed.append(True)

    convert = Mock()
    cleaner = Mock()
    monkeypatch.setattr(pipeline, "load_whisper_model", load_model)
    monkeypatch.setattr(pipeline, "convert_to_wav_16_mono", convert)
    monkeypatch.setattr(pipeline, "reduce_noise", cleaner)
    monkeypatch.setattr(pipeline, "create_name_trans_path", lambda name: output)
    assert pipeline.main([str(source), "out", "--language", language, "--noise-reduce"]) == 1
    convert.assert_not_called()
    cleaner.assert_not_called()
    model.transcribe.assert_not_called()
    assert closed == [True]
    assert source.is_file()
    assert not output.exists()
    assert repr(language) in caplog.text
    assert ", ".join(supported) in caplog.text
    assert all(record.exc_info is None for record in caplog.records)


@pytest.mark.parametrize("stage", ["call", "iterate"])
def test_library_language_rejection_is_reported_and_cleaned(
    tmp_path, make_wav, monkeypatch, caplog, stage
):
    source = make_wav(tmp_path / "source.wav")
    output = tmp_path / "out.md"
    failure = ValueError("'ja' is not a valid language code (accepted language codes: en)")
    model = Mock(supported_languages=["ja"])
    temporary = []
    closed = []

    def convert(path, directory):
        temporary.append(directory)
        return [make_wav(directory / "part.wav")]

    def segments():
        yield SimpleNamespace(start=0, end=1, text="partial")
        raise failure

    if stage == "call":
        model.transcribe.side_effect = failure
    else:
        model.transcribe.side_effect = lambda *args, **kwargs: (segments(), None)

    @contextmanager
    def load_model():
        try:
            yield model
        finally:
            closed.append(True)

    monkeypatch.setattr(pipeline, "load_whisper_model", load_model)
    monkeypatch.setattr(pipeline, "convert_to_wav_16_mono", convert)
    monkeypatch.setattr(pipeline, "create_name_trans_path", lambda name: output)
    with pytest.raises(pipeline.UnsupportedLanguageError) as caught:
        pipeline.transcribe_file(source, output, language="ja")
    assert caught.value.__cause__ is failure
    assert pipeline.main([str(source), "out", "--language", "ja"]) == 1
    assert "Código de idioma no aceptado por la biblioteca" in caplog.text
    assert all(record.exc_info is None for record in caplog.records)
    assert closed == [True, True]
    assert all(not directory.exists() for directory in temporary)
    assert source.is_file()
    assert not output.exists()
