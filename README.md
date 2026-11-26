# AI-Based Emotion Recognition in Azerbaijani and English Speech

This project compares speech-emotion recognition in Azerbaijani and English.
The first reproducible benchmark uses the RAVDESS English speech corpus and
speaker-independent validation. The Azerbaijani side is added through a CSV
manifest so that it can use a publicly available or explicitly licensed
dataset without treating downloaded files as ground truth.

## Initial scope

- **Main task:** classify each speech clip into a shared emotion category.
- **Baseline:** MFCC and prosodic features with logistic regression and SVM.
- **Primary metric:** macro-F1.
- **Validation:** held-out speakers selected before feature extraction.
- **Resource report:** model size, CPU inference time, and process memory.
- **Escalation:** optional and only enabled with conversation-level labels.

The repository intentionally does not generate escalation labels automatically
from emotion predictions. Such labels would not be clinically or scientifically
supported without a human-annotated conversation definition.

## Setup

Open PowerShell in the cloned repository root (the folder containing
`requirements.txt`), then run:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name emotion-project --display-name "Emotion Project"
```

## Download and inspect the English benchmark

```powershell
python src/download_ravdess.py
python src/train.py --language english --ravdess-dir data/raw/ravdess
```

RAVDESS is released for non-commercial academic use under
CC BY-NC-SA 4.0. Do not present it as a replacement for real Azerbaijani
customer-service recordings.

## Azerbaijani movie-audio pilot

The pilot WAV is kept locally at:

```text
data/raw/azerbaijani/source/Qaynana_1978_filmi.wav
```

The movie WAV and annotation CSVs are not included in the repository. The
Azerbaijani run is reproducible only when you have authorized access to the
audio and the corresponding labeled manifest. Without those files, use the
English RAVDESS benchmark for the demonstration.

Generate candidate speech timestamps and an annotation manifest:

```powershell
python src/segment_audio.py data/raw/azerbaijani/source/Qaynana_1978_filmi.wav
```

This writes `data/processed/azerbaijani/qaynana_segments.csv`. It stores
timestamps pointing into the source WAV and does not create separate audio
clips. By default it selects 300 time-spread candidates from all VAD regions;
use `--max-segments 0` to include all regions. WebRTC VAD candidates are not
verified speech or emotion labels. Review them and fill `speaker_id`,
`label_annotator_1`, `label_annotator_2`, and the adjudicated `label`. See
[ANNOTATION_GUIDE.md](ANNOTATION_GUIDE.md).

You do not need to hand-edit all 300 rows. A 50-segment time-spread pilot and
audio-player notebook are provided:

```powershell
python src/create_annotation_pilot.py
jupyter notebook notebooks/Annotate_Qaynana_Pilot.ipynb
```

In the notebook, set `annotator_id` to your own short name. If working alone,
one annotator is enough for the pilot; note this limitation in the report.
Listen to each segment, assign a stable speaker ID when possible, choose a
label, and review the segment. Changes are autosaved when you edit a field or
move to another segment; **Save** is also available for an explicit save. The
output is written to `data/processed/azerbaijani/labels_<annotator>.csv`.
The notebook also displays a live table of all segments and their current
saved labels below the audio player.
To check your own
consistency, optionally relabel a small random subset later without looking
at your first labels.

The manifest's `filepath` is just `Qaynana_1978_filmi.wav`; the loader resolves
the basename from `data/raw/azerbaijani/source/`.

Train after labels and speaker IDs are reviewed:

```powershell
python src/train.py --language azerbaijani --manifest data/processed/azerbaijani/labels_annotator_1.csv
```

Replace `annotator_1` with the annotator ID used for your CSV.

After both datasets have reviewed labels, run cross-language evaluation in
both directions:

```powershell
python src/cross_language.py --source english --target azerbaijani
python src/cross_language.py --source azerbaijani --target english
```

The runner evaluates only emotion labels present in both datasets and reports
per-class metrics and confusion matrices. It does not tune on the target
language.

For another Azerbaijani corpus, create a CSV with `filepath`, `label`, and
`speaker_id`. Segment-level manifests can also provide `start_seconds` and
`end_seconds`; both columns are required together. Blank labels are ignored.

Use the shared labels:

```text
neutral, calm, happy, sad, angry, fearful, disgust, surprised
```

These are acted performances. Do not present results from this pilot as
performance on natural customer-service conversations.

