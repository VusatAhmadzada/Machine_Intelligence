from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import webrtcvad


FRAME_MILLISECONDS = 30
VAD_SAMPLE_RATE = 16000
SUPPORTED_SAMPLE_RATES = {8000, 16000, 32000, 48000}
FRAMES_PER_BLOCK = 32


def speech_regions(
    flags: list[bool],
    duration_seconds: float,
    frame_seconds: float = FRAME_MILLISECONDS / 1000,
    merge_gap_seconds: float = 0.30,
    padding_seconds: float = 0.15,
    minimum_duration_seconds: float = 0.50,
    maximum_duration_seconds: float = 12.0,
) -> list[tuple[float, float]]:
    """Merge frame-level VAD flags into padded time regions."""
    speech_frames = [index for index, is_speech in enumerate(flags) if is_speech]
    if not speech_frames:
        return []

    max_gap = int(merge_gap_seconds / frame_seconds)
    padding = int(padding_seconds / frame_seconds)
    groups: list[list[int]] = [[speech_frames[0]]]

    for frame_index in speech_frames[1:]:
        if frame_index - groups[-1][-1] - 1 <= max_gap:
            groups[-1].append(frame_index)
        else:
            groups.append([frame_index])

    regions = []
    for group in groups:
        start = max(0, group[0] - padding) * frame_seconds
        end = min(len(flags), group[-1] + padding + 1) * frame_seconds
        end = min(end, duration_seconds)
        while end - start > maximum_duration_seconds:
            split_end = start + maximum_duration_seconds
            if split_end - start >= minimum_duration_seconds:
                regions.append((round(start, 3), round(split_end, 3)))
            start = split_end
        if end - start >= minimum_duration_seconds:
            regions.append((round(start, 3), round(end, 3)))
    return regions


def detect_speech(
    audio_path: Path,
    aggressiveness: int = 3,
) -> tuple[list[bool], float]:
    """Run WebRTC VAD on streaming mono frames without writing audio copies."""
    vad = webrtcvad.Vad(aggressiveness)
    flags: list[bool] = []
    with sf.SoundFile(audio_path) as source:
        sample_rate = source.samplerate
        if sample_rate not in SUPPORTED_SAMPLE_RATES:
            supported = ", ".join(
                str(rate) for rate in sorted(SUPPORTED_SAMPLE_RATES)
            )
            raise ValueError(
                f"Unsupported sample rate {sample_rate}; "
                f"expected one of {supported}"
            )
        frame_samples = sample_rate * FRAME_MILLISECONDS // 1000
        block_samples = frame_samples * FRAMES_PER_BLOCK
        duration = len(source) / sample_rate

        while True:
            audio = source.read(
                block_samples,
                dtype="float32",
                always_2d=True,
            )
            if audio.size == 0:
                break

            mono = audio.mean(axis=1)
            downsampled = resample_poly(
                mono,
                VAD_SAMPLE_RATE,
                sample_rate,
            )
            complete_frames = len(downsampled) // 480
            for frame_index in range(complete_frames):
                frame = downsampled[
                    frame_index * 480 : (frame_index + 1) * 480
                ]
                pcm = np.clip(frame * 32768, -32768, 32767).astype("<i2")
                flags.append(vad.is_speech(pcm.tobytes(), VAD_SAMPLE_RATE))

    return flags, duration


def write_manifest(
    audio_path: Path,
    output_path: Path,
    title: str,
    language: str,
    regions: list[tuple[float, float]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "segment_id",
        "filepath",
        "start_seconds",
        "end_seconds",
        "duration_seconds",
        "source_title",
        "language",
        "speaker_id",
        "label_annotator_1",
        "label_annotator_2",
        "label",
        "review_status",
        "notes",
    ]
    with output_path.open("w", newline="", encoding="utf-8-sig") as output:
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        for index, (start, end) in enumerate(regions, start=1):
            writer.writerow(
                {
                    "segment_id": f"segment_{index:05d}",
                    "filepath": str(audio_path.resolve()),
                    "start_seconds": start,
                    "end_seconds": end,
                    "duration_seconds": round(end - start, 3),
                    "source_title": title,
                    "language": language,
                    "speaker_id": "",
                    "label_annotator_1": "",
                    "label_annotator_2": "",
                    "label": "",
                    "review_status": "",
                    "notes": "",
                }
            )


def temporal_sample(
    regions: list[tuple[float, float]],
    maximum_segments: int,
) -> list[tuple[float, float]]:
    """Select a deterministic, time-spread subset of candidate regions."""
    if maximum_segments <= 0 or len(regions) <= maximum_segments:
        return regions
    indices = np.linspace(
        0,
        len(regions) - 1,
        num=maximum_segments,
        dtype=int,
    )
    return [regions[index] for index in indices]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a time-coded speech-candidate annotation manifest."
    )
    parser.add_argument("audio", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/azerbaijani/qaynana_segments.csv"),
    )
    parser.add_argument("--title", default="Qaynana (1978)")
    parser.add_argument("--language", default="azerbaijani")
    parser.add_argument("--aggressiveness", type=int, choices=range(4), default=3)
    parser.add_argument("--merge-gap", type=float, default=0.30)
    parser.add_argument("--padding", type=float, default=0.15)
    parser.add_argument("--minimum-duration", type=float, default=0.50)
    parser.add_argument("--maximum-duration", type=float, default=12.0)
    parser.add_argument(
        "--max-segments",
        type=int,
        default=300,
        help="Review sample size; use 0 to include every candidate.",
    )
    args = parser.parse_args()

    if not args.audio.is_file():
        raise FileNotFoundError(f"Audio file not found: {args.audio}")
    if args.maximum_duration <= 0 or args.minimum_duration < 0:
        raise ValueError("Segment duration limits must be positive")
    if args.max_segments < 0:
        raise ValueError("--max-segments must be zero or a positive integer")

    flags, duration = detect_speech(args.audio, args.aggressiveness)
    regions = speech_regions(
        flags,
        duration,
        merge_gap_seconds=args.merge_gap,
        padding_seconds=args.padding,
        minimum_duration_seconds=args.minimum_duration,
        maximum_duration_seconds=args.maximum_duration,
    )
    detected_count = len(regions)
    regions = temporal_sample(regions, args.max_segments)
    write_manifest(
        args.audio,
        args.output,
        args.title,
        args.language,
        regions,
    )

    speech_duration = sum(end - start for start, end in regions)
    print(f"Audio duration: {duration / 60:.1f} minutes")
    print(f"Candidate speech regions found: {detected_count}")
    print(f"Regions selected for review: {len(regions)}")
    print(f"Selected review duration: {speech_duration / 60:.1f} minutes")
    print(f"Manifest: {args.output}")
    print("VAD regions require manual review; they are not emotion labels.")


if __name__ == "__main__":
    main()
