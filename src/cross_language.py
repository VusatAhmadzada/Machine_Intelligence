from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from train import SHARED_LABELS, extract_matrix, load_manifest, load_ravdess


def load_language_data(
    language: str,
    ravdess_dir: Path,
    manifest_path: Path,
) -> pd.DataFrame:
    if language == "english":
        return load_ravdess(ravdess_dir)
    if language == "azerbaijani":
        return load_manifest(manifest_path)
    raise ValueError(f"Unsupported language: {language}")


def shared_labels(source: pd.DataFrame, target: pd.DataFrame) -> list[str]:
    source_labels = set(source["label"].astype(str))
    target_labels = set(target["label"].astype(str))
    return [
        label
        for label in SHARED_LABELS
        if label in source_labels and label in target_labels
    ]


def evaluate_direction(
    source_language: str,
    target_language: str,
    source: pd.DataFrame,
    target: pd.DataFrame,
    output_dir: Path,
) -> dict:
    labels = shared_labels(source, target)
    if len(labels) < 2:
        raise ValueError(
            "At least two shared, human-verified emotion labels are required"
        )

    source_features = extract_matrix(source)
    target_features = extract_matrix(target)
    source_mask = source["label"].astype(str).isin(labels).to_numpy()
    target_mask = target["label"].astype(str).isin(labels).to_numpy()
    x_train = source_features[source_mask]
    y_train = source.loc[source_mask, "label"].astype(str)
    x_test = target_features[target_mask]
    y_test = target.loc[target_mask, "label"].astype(str)

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

    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        start = time.perf_counter()
        predictions = model.predict(x_test)
        inference_ms = (time.perf_counter() - start) * 1000
        model_path = output_dir / (
            f"{source_language}_to_{target_language}_{name}.joblib"
        )
        joblib.dump(model, model_path)
        results[name] = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "macro_f1": float(
                f1_score(
                    y_test,
                    predictions,
                    labels=labels,
                    average="macro",
                    zero_division=0,
                )
            ),
            "model_size_bytes": model_path.stat().st_size,
            "target_inference_ms": inference_ms,
            "confusion_matrix": confusion_matrix(
                y_test,
                predictions,
                labels=labels,
            ).tolist(),
            "classification_report": classification_report(
                y_test,
                predictions,
                labels=labels,
                zero_division=0,
                output_dict=True,
            ),
        }

    report = {
        "source_language": source_language,
        "target_language": target_language,
        "shared_labels": labels,
        "source_clips": int(len(y_train)),
        "target_clips": int(len(y_test)),
        "source_speakers": int(
            source.loc[source_mask, "speaker_id"].nunique()
        ),
        "target_speakers": int(
            target.loc[target_mask, "speaker_id"].nunique()
        ),
        "methods": results,
        "caveat": (
            "Cross-dataset results depend on acting style, recording "
            "conditions, and annotation agreement; they do not establish "
            "real-world call-center performance."
        ),
    }
    report_path = output_dir / f"{source_language}_to_{target_language}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["english", "azerbaijani"], required=True)
    parser.add_argument("--target", choices=["english", "azerbaijani"], required=True)
    parser.add_argument(
        "--ravdess-dir",
        type=Path,
        default=Path("data/raw/ravdess"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/processed/azerbaijani/qaynana_segments.csv"),
    )
    parser.add_argument("--output", type=Path, default=Path("results/cross_language"))
    args = parser.parse_args()
    if args.source == args.target:
        raise ValueError("Source and target languages must be different")

    source = load_language_data(
        args.source,
        args.ravdess_dir,
        args.manifest,
    )
    target = load_language_data(
        args.target,
        args.ravdess_dir,
        args.manifest,
    )
    report = evaluate_direction(
        args.source,
        args.target,
        source,
        target,
        args.output,
    )
    for method, metrics in report["methods"].items():
        print(
            f"{method}: accuracy={metrics['accuracy']:.3f}, "
            f"macro-F1={metrics['macro_f1']:.3f}"
        )


if __name__ == "__main__":
    main()
