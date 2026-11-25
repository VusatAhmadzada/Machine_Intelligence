from __future__ import annotations

from pathlib import Path

import ipywidgets as widgets
import numpy as np
import pandas as pd
import soundfile as sf
from IPython.display import Audio, clear_output, display

from train import resolve_manifest_filepath


EMOTION_OPTIONS = [
    "",
    "neutral",
    "happy",
    "sad",
    "angry",
    "unclear",
]
REVIEW_OPTIONS = ["unreviewed", "keep", "exclude"]
ANNOTATION_COLUMNS = ["speaker_id", "label", "review_status", "notes"]


def annotation_values(
    selected_label: str,
    selected_status: str,
    speaker_id: str,
    notes: str,
) -> dict[str, str]:
    excluded = selected_status == "exclude" or selected_label == "unclear"
    return {
        "speaker_id": "" if excluded else speaker_id.strip(),
        "label": (
            "" if excluded or not selected_label else selected_label
        ),
        "review_status": "exclude" if excluded else selected_status,
        "notes": notes.strip(),
    }


def _read_segment(row: pd.Series, manifest_path: Path) -> tuple[np.ndarray, int]:
    filepath = resolve_manifest_filepath(
        str(row["filepath"]),
        manifest_path,
        str(row.get("language", "azerbaijani")),
    )
    start = float(row["start_seconds"])
    end = float(row["end_seconds"])
    with sf.SoundFile(filepath) as audio_file:
        sample_rate = audio_file.samplerate
        audio_file.seek(round(start * sample_rate))
        samples = audio_file.read(
            round((end - start) * sample_rate),
            dtype="float32",
            always_2d=True,
        )
    if samples.size == 0:
        raise ValueError(f"No audio samples found for {row['segment_id']}")
    mono = samples.mean(axis=1)
    return mono, sample_rate


def launch_annotation_ui(
    manifest_path: Path,
    output_path: Path,
) -> None:
    """Display an audio player and autosave one annotator's pilot labels."""
    manifest_path = Path(manifest_path)
    output_path = Path(output_path)
    frame = pd.read_csv(manifest_path)
    required = {
        "segment_id",
        "filepath",
        "start_seconds",
        "end_seconds",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Annotation manifest has no segments")

    for column in ANNOTATION_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""

    if output_path.exists():
        existing = pd.read_csv(output_path).set_index("segment_id")
        for column in ANNOTATION_COLUMNS:
            if column in existing.columns:
                saved = existing[column].to_dict()
                frame[column] = frame["segment_id"].map(saved).fillna(
                    frame[column]
                )

    for column in ANNOTATION_COLUMNS:
        frame[column] = frame[column].astype("string").fillna("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False, encoding="utf-8-sig")

    segment_index = widgets.IntSlider(
        value=0,
        min=0,
        max=len(frame) - 1,
        step=1,
        description="Segment",
        continuous_update=False,
        style={"description_width": "initial"},
    )
    speaker_id = widgets.Text(
        description="Speaker ID",
        placeholder="e.g. speaker_01; reuse for the same voice",
        style={"description_width": "initial"},
        layout=widgets.Layout(width="95%"),
    )
    label = widgets.Dropdown(
        options=EMOTION_OPTIONS,
        description="Voice emotion",
        style={"description_width": "initial"},
    )
    review_status = widgets.Dropdown(
        options=REVIEW_OPTIONS,
        value="unreviewed",
        description="Review",
        style={"description_width": "initial"},
    )
    notes = widgets.Text(
        description="Notes",
        placeholder="Optional",
        style={"description_width": "initial"},
        layout=widgets.Layout(width="95%"),
    )
    previous_button = widgets.Button(description="Previous")
    next_button = widgets.Button(description="Next")
    save_button = widgets.Button(
        description="Save",
        button_style="success",
        tooltip="Save this segment to your separate annotation CSV",
    )
    message = widgets.HTML()
    audio_output = widgets.Output()
    annotation_table = widgets.HTML()
    active_index: list[int | None] = [None]
    loading_row = [False]

    def refresh_annotation_table() -> None:
        columns = ["segment_id", *ANNOTATION_COLUMNS]
        table = frame[columns].to_html(
            index=False,
            border=0,
            classes="annotation-table",
        )
        annotation_table.value = (
            "<div style='max-height:360px; overflow:auto; "
            "border:1px solid #aaa; padding:6px'>"
            "<b>All segments — updates as they are autosaved</b>"
            f"{table}</div>"
        )

    def show_segment(index: int) -> None:
        row = frame.iloc[index]
        start = float(row["start_seconds"])
        end = float(row["end_seconds"])
        with audio_output:
            clear_output(wait=True)
            print(
                f"{row['segment_id']} | {start:.2f}s–{end:.2f}s "
                f"({end - start:.2f}s)"
            )
            samples, sample_rate = _read_segment(row, manifest_path)
            display(Audio(samples, rate=sample_rate))

    def persist_current(status: str) -> None:
        index = active_index[0]
        if index is None or loading_row[0]:
            return
        values = annotation_values(
            label.value,
            review_status.value,
            speaker_id.value,
            notes.value,
        )
        for column, value in values.items():
            frame.loc[index, column] = value
        frame.to_csv(output_path, index=False, encoding="utf-8-sig")
        refresh_annotation_table()
        message.value = (
            f"<b>{status} {frame.iloc[index]['segment_id']}</b> to "
            f"<code>{output_path.name}</code>"
        )

    def load_row(change: dict | None = None) -> None:
        if active_index[0] is not None:
            persist_current("Auto-saved")
        active_index[0] = segment_index.value
        loading_row[0] = True
        row = frame.iloc[active_index[0]]
        speaker_id.value = (
            "" if pd.isna(row["speaker_id"]) else str(row["speaker_id"])
        )
        current_label = "" if pd.isna(row["label"]) else str(row["label"])
        label.value = current_label if current_label in EMOTION_OPTIONS else ""
        current_status = (
            "unreviewed"
            if pd.isna(row["review_status"]) or not row["review_status"]
            else str(row["review_status"])
        )
        review_status.value = (
            current_status
            if current_status in REVIEW_OPTIONS
            else "unreviewed"
        )
        notes.value = "" if pd.isna(row["notes"]) else str(row["notes"])
        loading_row[0] = False
        message.value = (
            f"<b>{active_index[0] + 1} of {len(frame)}</b>"
        )
        show_segment(active_index[0])

    def save_row(_: widgets.Button) -> None:
        persist_current("Saved")

    def move_previous(_: widgets.Button) -> None:
        segment_index.value = max(segment_index.min, segment_index.value - 1)

    def move_next(_: widgets.Button) -> None:
        segment_index.value = min(segment_index.max, segment_index.value + 1)

    segment_index.observe(load_row, names="value")
    for control in (speaker_id, label, review_status, notes):
        control.observe(
            lambda change: persist_current("Auto-saved"),
            names="value",
        )
    save_button.on_click(save_row)
    previous_button.on_click(move_previous)
    next_button.on_click(move_next)

    controls = widgets.VBox(
        [
            segment_index,
            speaker_id,
            label,
            review_status,
            notes,
            widgets.HBox([previous_button, save_button, next_button]),
            message,
        ]
    )
    refresh_annotation_table()
    interface = widgets.VBox([controls, audio_output, annotation_table])
    load_row()
    display(interface)
