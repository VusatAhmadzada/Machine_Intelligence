from src.segment_audio import speech_regions, temporal_sample


def test_speech_regions_merge_short_gaps_and_apply_padding():
    flags = [False] * 20
    flags[3:6] = [True] * 3
    flags[8:11] = [True] * 3

    regions = speech_regions(
        flags,
        duration_seconds=0.6,
        merge_gap_seconds=0.1,
        padding_seconds=0.03,
        minimum_duration_seconds=0.1,
    )

    assert regions == [(0.06, 0.36)]


def test_speech_regions_drop_short_noise_bursts():
    flags = [False] * 20
    flags[5] = True

    assert speech_regions(
        flags,
        duration_seconds=0.6,
        padding_seconds=0,
        minimum_duration_seconds=0.1,
    ) == []


def test_temporal_sample_is_deterministic_and_spans_timeline():
    regions = [(float(index), float(index + 1)) for index in range(10)]

    sample = temporal_sample(regions, 4)

    assert sample == [regions[0], regions[3], regions[6], regions[9]]


def test_zero_temporal_sample_limit_keeps_all_regions():
    regions = [(0.0, 1.0), (2.0, 3.0)]

    assert temporal_sample(regions, 0) == regions


def test_speech_regions_split_long_continuous_regions():
    flags = [True] * 100

    regions = speech_regions(
        flags,
        duration_seconds=3.0,
        padding_seconds=0,
        minimum_duration_seconds=0.1,
        maximum_duration_seconds=1.0,
    )

    assert regions == [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)]
