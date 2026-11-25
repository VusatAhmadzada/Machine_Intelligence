import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from train import load_manifest, resolve_manifest_filepath


def test_load_manifest_skips_unlabeled_candidates_and_keeps_segment_times(
    tmp_path,
):
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            {
                "filepath": "movie.wav",
                "start_seconds": 1.0,
                "end_seconds": 2.5,
                "speaker_id": "speaker_01",
                "label": " Angry ",
            },
            {
                "filepath": "movie.wav",
                "start_seconds": 3.0,
                "end_seconds": 4.0,
                "speaker_id": "",
                "label": "",
            },
        ]
    ).to_csv(manifest, index=False)

    frame = load_manifest(manifest)

    assert len(frame) == 1
    assert frame.iloc[0]["label"] == "angry"
    assert frame.iloc[0]["start_seconds"] == 1.0
    assert frame.iloc[0]["end_seconds"] == 2.5


def test_load_manifest_requires_both_segment_time_columns(tmp_path):
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            {
                "filepath": "movie.wav",
                "start_seconds": 1.0,
                "label": "angry",
                "speaker_id": "speaker_01",
            }
        ]
    ).to_csv(manifest, index=False)

    with pytest.raises(ValueError, match="both start_seconds and end_seconds"):
        load_manifest(manifest)


def test_resolve_manifest_filepath_accepts_filename_only(tmp_path):
    root = tmp_path / "project"
    source = root / "data" / "raw" / "azerbaijani" / "source"
    manifest = root / "data" / "processed" / "azerbaijani" / "manifest.csv"
    source.mkdir(parents=True)
    manifest.parent.mkdir(parents=True)
    audio = source / "Qaynana.wav"
    audio.touch()

    resolved = resolve_manifest_filepath(
        "Qaynana.wav",
        manifest,
        "azerbaijani",
        project_root=root,
    )

    assert Path(resolved) == audio
