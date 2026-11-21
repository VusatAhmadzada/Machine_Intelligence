import numpy as np
import pytest
import soundfile as sf

from src.features import extract_features


def test_extract_features_reads_only_requested_segment(tmp_path):
    sample_rate = 16000
    time = np.arange(sample_rate * 2) / sample_rate
    audio = 0.1 * np.sin(2 * np.pi * 220 * time)
    path = tmp_path / "tone.wav"
    sf.write(path, audio, sample_rate)

    features = extract_features(
        str(path),
        start_seconds=0.5,
        end_seconds=1.5,
    )

    assert features.shape == (86,)
    assert np.isfinite(features).all()


def test_extract_features_rejects_incomplete_segment_range():
    with pytest.raises(ValueError, match="Both segment start and end"):
        extract_features("unused.wav", start_seconds=0.5)
