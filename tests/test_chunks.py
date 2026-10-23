from processor.chunks import build_audio_chunks


def test_offsets_use_actual_durations(tmp_path, make_wav):
    paths = [make_wav(tmp_path / "a.wav", 2.25), make_wav(tmp_path / "b.wav", 1.5)]
    chunks = build_audio_chunks(paths)
    assert [chunk.path for chunk in chunks] == paths
    assert [chunk.offset_seconds for chunk in chunks] == [0, 2.25]
