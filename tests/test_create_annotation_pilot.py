import pandas as pd

from src.create_annotation_pilot import create_pilot


def test_create_pilot_is_time_spread_and_preserves_rows(tmp_path):
    source = tmp_path / "source.csv"
    output = tmp_path / "pilot.csv"
    pd.DataFrame(
        {
            "segment_id": [f"segment_{index}" for index in range(10)],
            "start_seconds": list(range(10)),
        }
    ).to_csv(source, index=False)

    pilot = create_pilot(source, output, sample_size=4)

    assert pilot["segment_id"].tolist() == [
        "segment_0",
        "segment_3",
        "segment_6",
        "segment_9",
    ]
    assert len(pd.read_csv(output)) == 4
