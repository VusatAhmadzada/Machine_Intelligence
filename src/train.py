from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import tracemalloc

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm

from features import extract_features
from ravdess import parse_ravdess_filename


SHARED_LABELS = [
    "neutral",
    "calm",
    "happy",
    "sad",
    "angry",
    "fearful",
    "disgust",
    "surprised",
]


def load_ravdess(root: Path) -> pd.DataFrame:
    records = []
    for wav in root.rglob("*.wav"):
        record = parse_ravdess_filename(wav)
        if record:
            records.append(record)
    if not records:
        raise FileNotFoundError(
            f"No RAVDESS speech recordings found below {root}"
        )
    frame = pd.DataFrame(records)
    frame["label"] = pd.Categorical(frame["label"], categories=SHARED_LABELS)
    return frame


def load_manifest(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"filepath", "label", "speaker_id"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")
    if ("start_seconds" in frame.columns) != ("end_seconds" in frame.columns):
        raise ValueError(
            "Segment manifests must include both start_seconds and end_seconds"
        )
    if "conversation_id" not in frame.columns:
        frame["conversation_id"] = ""
    frame["label"] = frame["label"].astype("string").str.strip().str.lower()
    frame = frame[
        frame["label"].notna() & frame["label"].ne("")
    ].copy()
    if frame.empty:
        raise ValueError(
            "Manifest has no labeled rows; annotate candidate segments first"
        )
    frame["speaker_id"] = (
        frame["speaker_id"].astype("string").str.strip()
    )
    if frame["speaker_id"].isna().any() or frame["speaker_id"].eq("").any():
        raise ValueError("Every labeled row must have a speaker_id")
    frame["filepath"] = frame["filepath"].map(
        lambda value: str(path.parent / value)
        if not Path(value).is_absolute()
        else value
    )
    invalid = set(frame["label"].unique()) - set(SHARED_LABELS)
    if invalid:
        raise ValueError(f"Unknown labels: {sorted(invalid)}")
    if {"start_seconds", "end_seconds"}.issubset(frame.columns):
        frame["start_seconds"] = pd.to_numeric(
            frame["start_seconds"],
            errors="raise",
        )
        frame["end_seconds"] = pd.to_numeric(
            frame["end_seconds"],
            errors="raise",
        )
        if (
            frame["start_seconds"].lt(0).any()
            or frame["end_seconds"].le(frame["start_seconds"]).any()
        ):
            raise ValueError("Manifest contains invalid segment time ranges")
    frame["language"] = path.parent.name
    return frame


def speaker_split(
    frame: pd.DataFrame,
    test_size: float = 0.25,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    if frame["speaker_id"].nunique() < 2:
        raise ValueError("At least two speakers are needed for a speaker split")
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )
    train, test = next(
        splitter.split(frame, groups=frame["speaker_id"])
    )
    return train, test


def extract_matrix(frame: pd.DataFrame) -> np.ndarray:
    vectors = []
    for _, row in tqdm(frame.iterrows(), total=len(frame), desc="Extracting features"):
        start = row.get("start_seconds")
        end = row.get("end_seconds")
        segment_options = {}
        if pd.notna(start) and pd.notna(end):
            segment_options = {
                "start_seconds": float(start),
                "end_seconds": float(end),
            }
        vectors.append(
            extract_features(row["filepath"], **segment_options)
        )
    matrix = np.vstack(vectors)
    if not np.isfinite(matrix).all():
        raise ValueError("Feature matrix contains NaN or infinity")
    return matrix


def train_and_evaluate(
    frame: pd.DataFrame,
    features: np.ndarray,
    output_dir: Path,
) -> dict:
    train_idx, test_idx = speaker_split(frame)
    y_train = frame.iloc[train_idx]["label"].astype(str)
    y_test = frame.iloc[test_idx]["label"].astype(str)

    models = {
        "majority": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                random_state=42,
            ),
        ),
        "rbf_svm": make_pipeline(
            StandardScaler(),
            SVC(
                kernel="rbf",
                class_weight="balanced",
                C=10.0,
            ),
        ),
    }

    results = {}
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, model in models.items():
        model.fit(features[train_idx], y_train)
        predictions = model.predict(features[test_idx])
        macro_f1 = f1_score(y_test, predictions, average="macro")
        accuracy = accuracy_score(y_test, predictions)

        tracemalloc.start()
        start = time.perf_counter()
        model.predict(features[test_idx[: min(100, len(test_idx))]])
        elapsed = (time.perf_counter() - start) * 1000
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        model_path = output_dir / f"{name}.joblib"
        joblib.dump(model, model_path)
        results[name] = {
            "accuracy": float(accuracy),
            "macro_f1": float(macro_f1),
            "model_size_bytes": model_path.stat().st_size,
            "cpu_inference_ms_for_100_or_fewer_clips": float(elapsed),
            "sklearn_traced_peak_bytes": peak_bytes,
            "train_speaker_count": int(frame.iloc[train_idx]["speaker_id"].nunique()),
            "test_speaker_count": int(frame.iloc[test_idx]["speaker_id"].nunique()),
            "confusion_matrix": confusion_matrix(
                y_test,
                predictions,
                labels=SHARED_LABELS,
            ).tolist(),
            "classification_report": classification_report(
                y_test,
                predictions,
                labels=SHARED_LABELS,
                zero_division=0,
                output_dict=True,
            ),
        }
        print(
            f"{name}: accuracy={accuracy:.3f}, macro-F1={macro_f1:.3f}"
        )

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", choices=["english", "azerbaijani"])
    parser.add_argument(
        "--ravdess-dir",
        type=Path,
        default=Path("data/raw/ravdess"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/raw/azerbaijani/manifest.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results"),
    )
    args = parser.parse_args()

    if args.language == "english":
        frame = load_ravdess(args.ravdess_dir)
    elif args.language == "azerbaijani":
        frame = load_manifest(args.manifest)
    else:
        raise ValueError("Choose --language english or --language azerbaijani")

    print(
        f"{len(frame)} clips, "
        f"{frame['speaker_id'].nunique()} speakers"
    )
    print(frame["label"].value_counts().sort_index().to_string())
    features = extract_matrix(frame)
    train_and_evaluate(
        frame,
        features,
        args.output / args.language,
    )


if __name__ == "__main__":
    main()
