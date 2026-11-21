import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from cross_language import shared_labels


def test_shared_labels_returns_only_common_labels_in_project_order():
    source = pd.DataFrame({"label": ["angry", "neutral", "happy", "calm"]})
    target = pd.DataFrame({"label": ["happy", "angry", "calm"]})

    assert shared_labels(source, target) == ["calm", "happy", "angry"]
