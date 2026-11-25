# Azerbaijani movie-audio annotation guide

The VAD manifest contains only candidate speech-like time ranges. Review each
candidate by listening to the source WAV at its start and end timestamps.

For each segment:

1. Mark it usable only when Azerbaijani speech is audible and one speaker is
   reasonably clear.
2. Assign a stable anonymous `speaker_id` to each voice and reuse it across
   the movie. Leave it blank if identity is uncertain.
3. Have annotators label the perceived vocal emotion independently in
   `label_annotator_1` and `label_annotator_2`.
4. Put the agreed/adjudicated label in `label`. If the annotators disagree and
   the team cannot resolve it, leave `label` blank.
5. Set `review_status` to `keep` or `exclude`; note music, overlap, unclear
   speech, or uncertainty in `notes`.

If working alone, use one annotator file and describe the labels as
single-annotator annotations. If practical, revisit a random subset after a
few days without looking at your earlier labels and report self-agreement as a
consistency check. This does not replace independent annotators.

Allowed labels:

```text
neutral, calm, happy, sad, angry, fearful, disgust, surprised
```

Label the voice, not the plot or subtitles. Exclude segments with music
masking speech, multiple indistinguishable speakers, unintelligible speech, or
unclear emotion. These are acted performances, not natural customer-service
calls.

Keep all segments from a speaker together when splitting training and testing.
Results from one movie are a source-specific pilot, not evidence of
generalization to other films or real conversations.
