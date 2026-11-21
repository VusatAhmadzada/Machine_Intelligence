from __future__ import annotations

from pathlib import Path
import re

import pandas as pd


EMOTION_NAMES = {
    1: "neutral",
    2: "calm",
    3: "happy",
    4: "sad",
    5: "angry",
    6: "fearful",
    7: "disgust",
    8: "surprised",
}

RAVDESS_FILENAME = re.compile(
    r"^(?P<modality>\d{2})-(?P<channel>\d{2})-(?P<emotion>\d{2})-"
    r"(?P<intensity>\d{2})-(?P<statement>\d{2})-(?P<repetition>\d{2})-"
    r"(?P<actor>\d{2})\.wav$"
)


def parse_ravdess_filename(path: str | Path) -> dict[str, int | str] | None:
    """Parse a RAVDESS audio-only filename."""
    match = RAVDESS_FILENAME.match(Path(path).name)
    if not match:
        return None

    parts = {name: int(value) for name, value in match.groupdict().items()}
    emotion_id = parts["emotion"]
    if emotion_id not in EMOTION_NAMES:
        return None
    return {
        "filepath": str(path),
        "label": EMOTION_NAMES[emotion_id],
        "speaker_id": f"actor_{parts['actor']:02d}",
        "conversation_id": "",
        "language": "english",
        "intensity": parts["intensity"],
        "statement": parts["statement"],
        "repetition": parts["repetition"],
    }

