from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def create_pilot(
    manifest_path: Path,
    output_path: Path,
    sample_size: int = 50,
) -> pd.DataFrame:
    if sample_size < 1:
        raise ValueError("Sample size must be at least one")
    frame = pd.read_csv(manifest_path)
    if frame.empty:
        raise ValueError(f"Manifest is empty: {manifest_path}")
    sample_size = min(sample_size, len(frame))
    indices = np.linspace(
        0,
        len(frame) - 1,
        num=sample_size,
        dtype=int,
    )
    pilot = frame.iloc[indices].copy().reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pilot.to_csv(output_path, index=False, encoding="utf-8-sig")
    return pilot


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a time-spread human annotation pilot."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/processed/azerbaijani/qaynana_segments.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/azerbaijani/qaynana_pilot_50.csv"),
    )
    parser.add_argument("--sample-size", type=int, default=50)
    args = parser.parse_args()
    pilot = create_pilot(args.manifest, args.output, args.sample_size)
    print(f"Created {len(pilot)} review rows at {args.output}")


if __name__ == "__main__":
    main()
