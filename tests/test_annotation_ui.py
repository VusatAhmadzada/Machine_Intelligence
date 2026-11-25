import sys
from pathlib import Path

import ipywidgets as widgets
import numpy as np
import pandas as pd
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from src import annotation_ui
from src.annotation_ui import annotation_values


def test_unclear_annotation_is_saved_as_excluded_without_a_label():
    values = annotation_values("unclear", "keep", "speaker_01", "music")

    assert values == {
        "speaker_id": "",
        "label": "",
        "review_status": "exclude",
        "notes": "music",
    }


def test_usable_annotation_keeps_label_and_normalizes_fields():
    values = annotation_values("angry", "keep", " speaker_01 ", "  clear  ")

    assert values == {
        "speaker_id": "speaker_01",
        "label": "angry",
        "review_status": "keep",
        "notes": "clear",
    }


def test_annotation_ui_creates_csv_and_autosaves_widget_changes(
    tmp_path,
    monkeypatch,
):
    sample_rate = 16000
    audio_path = tmp_path / "source.wav"
    sf.write(audio_path, np.full(sample_rate, 0.1, dtype=np.float32), sample_rate)
    manifest_path = tmp_path / "pilot.csv"
    pd.DataFrame(
        [
            {
                "segment_id": "segment_00001",
                "filepath": str(audio_path),
                "start_seconds": 0.0,
                "end_seconds": 1.0,
                "speaker_id": "",
                "label": "",
                "review_status": "",
                "notes": "",
            }
        ]
    ).to_csv(manifest_path, index=False)
    output_path = tmp_path / "labels.csv"
    displayed = []
    monkeypatch.setattr(annotation_ui, "display", displayed.append)

    annotation_ui.launch_annotation_ui(manifest_path, output_path)

    assert output_path.exists()
    interface = next(
        widget
        for widget in displayed
        if isinstance(widget, widgets.VBox)
    )

    def descendants(widget):
        yield widget
        for child in getattr(widget, "children", ()):
            yield from descendants(child)

    children = list(descendants(interface))
    label = next(
        widget
        for widget in children
        if isinstance(widget, widgets.Dropdown)
        and widget.description == "Voice emotion"
    )
    status = next(
        widget
        for widget in children
        if isinstance(widget, widgets.Dropdown)
        and widget.description == "Review"
    )
    annotation_table = next(
        widget
        for widget in children
        if isinstance(widget, widgets.HTML) and "<table" in widget.value
    )
    save_button = next(
        widget
        for widget in children
        if isinstance(widget, widgets.Button)
        and widget.description == "Save"
    )

    label.value = "angry"
    status.value = "keep"
    saved = pd.read_csv(output_path).iloc[0]
    assert saved["label"] == "angry"
    assert saved["review_status"] == "keep"
    assert "segment_00001" in annotation_table.value
    assert "angry" in annotation_table.value

    save_button.click()
    saved = pd.read_csv(output_path).iloc[0]
    assert saved["label"] == "angry"
