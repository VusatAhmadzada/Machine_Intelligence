# Dataset acquisition plan

## Confirmed English benchmark

The project includes RAVDESS as the initial English benchmark:

- 1440 speech clips
- 24 professional actors
- Audio-only `.wav` recordings
- 8 emotion categories
- CC BY-NC-SA 4.0 license
- Speaker-independent split

RAVDESS is acted speech. It is useful for a controlled baseline, but it does
not by itself prove performance in real Azerbaijani or English calls.

## Azerbaijani requirement

No verified public Azerbaijani speech-emotion dataset was found in the
public repositories searched for this project.

The current pilot source is the user-provided `Qaynana_1978_filmi.wav` file.
It is a 72-minute, stereo, 48 kHz WAV. The VAD script found 1,085 candidate
speech regions and selected 300 time-spread regions for human review. These are
only VAD candidates: they still need speech-quality checks, two independent
emotion labels, consensus labels, and stable speaker IDs.

This is acted movie speech, so results are limited to this source. It does not
provide evidence about natural customer-service conversations or escalation.

## Required dataset metadata

The team should provide:

- Dataset name and source link
- License or explicit research permission
- Clip and speaker counts
- Emotion labels and label definitions
- Recording language and regional accent
- Sampling rate
- Speaker identifier
- Conversation identifier, if available
- Existing train/test split or a way to recreate one

## Minimum label map

The comparison uses these shared labels:

```text
neutral
calm
happy
sad
angry
fearful
disgust
surprised
```

Do not merge labels silently. Record every label mapping in the manifest or
data-preparation code and keep the original label in a separate column.

## Current next step

Review `data/processed/azerbaijani/qaynana_segments.csv` using
[ANNOTATION_GUIDE.md](ANNOTATION_GUIDE.md). Fill labels and speaker IDs before
running Azerbaijani or cross-language model evaluations.

