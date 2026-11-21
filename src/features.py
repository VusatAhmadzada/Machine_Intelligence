from __future__ import annotations

import numpy as np
import librosa


def extract_features(
    audio_path: str,
    sample_rate: int = 16000,
    n_mfcc: int = 40,
    start_seconds: float | None = None,
    end_seconds: float | None = None,
) -> np.ndarray:
    """Create a fixed-length MFCC/prosody feature vector from one clip."""
    if (start_seconds is None) != (end_seconds is None):
        raise ValueError("Both segment start and end times must be provided")
    if start_seconds is not None and (
        start_seconds < 0
        or end_seconds is None
        or end_seconds <= start_seconds
    ):
        raise ValueError("Segment end must be greater than a nonnegative start")

    load_options = {}
    if start_seconds is not None:
        load_options = {
            "offset": start_seconds,
            "duration": end_seconds - start_seconds,
        }
    audio, source_rate = librosa.load(
        audio_path,
        sr=sample_rate,
        mono=True,
        **load_options,
    )
    if audio.size < int(source_rate * 0.05):
        raise ValueError(f"Clip is too short: {audio_path}")

    audio, _ = librosa.effects.trim(audio, top_db=30)
    if audio.size < int(source_rate * 0.05):
        raise ValueError(f"Clip contains no usable audio: {audio_path}")

    mfcc = librosa.feature.mfcc(y=audio, sr=source_rate, n_mfcc=n_mfcc)
    energy = librosa.feature.rms(y=audio).ravel()
    zcr = librosa.feature.zero_crossing_rate(audio).ravel()
    centroid = librosa.feature.spectral_centroid(
        y=audio,
        sr=source_rate,
    ).ravel()

    mfcc_features = np.concatenate(
        [mfcc.mean(axis=1), mfcc.std(axis=1)]
    )
    summary_features = np.array(
        [
            energy.mean(),
            energy.std(),
            zcr.mean(),
            zcr.std(),
            centroid.mean(),
            centroid.std(),
        ],
        dtype=np.float32,
    )
    return np.concatenate([mfcc_features, summary_features]).astype(np.float32)
