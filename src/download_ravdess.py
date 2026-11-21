from __future__ import annotations

import argparse
from pathlib import Path
import zipfile

import requests


URL = (
    "https://huggingface.co/datasets/jiwaz123/Speech-emotion-english/"
    "resolve/main/Audio_Speech_Actors_01-24.zip"
)


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        print(f"Using existing archive: {destination}")
        return

    print(f"Downloading RAVDESS: {url}")
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        downloaded = 0
        with destination.open("wb") as output:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                output.write(chunk)
                downloaded += len(chunk)
                if total:
                    print(
                        f"\r{downloaded / total:.1%}",
                        end="",
                        flush=True,
                    )
    print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/ravdess"),
    )
    args = parser.parse_args()

    archive = args.output.parent / "ravdess.zip"
    download(URL, archive)

    args.output.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zipped:
        if not zipped.namelist():
            raise ValueError("Downloaded RAVDESS archive is empty")
        zipped.extractall(args.output)

    wav_count = sum(1 for path in args.output.rglob("*.wav"))
    print(f"Extracted {wav_count} RAVDESS speech recordings")


if __name__ == "__main__":
    main()
