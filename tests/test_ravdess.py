from src.ravdess import parse_ravdess_filename


def test_parse_ravdess_filename():
    record = parse_ravdess_filename(
        "Actor_12/03-01-05-02-02-01-12.wav"
    )
    assert record is not None
    assert record["label"] == "angry"
    assert record["speaker_id"] == "actor_12"
    assert record["intensity"] == 2


def test_reject_unknown_ravdess_filename():
    assert parse_ravdess_filename("not-a-ravdess-file.wav") is None
