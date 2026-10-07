import importlib.util
from pathlib import Path

import pandas as pd
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_multi_context_improvement.py"
SPEC = importlib.util.spec_from_file_location("multi_context_followup", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_phone_cleaning_and_classes():
    assert MODULE.clean_phone("AY1") == "AY"
    assert MODULE.phone_class("AY1") == "vowel"
    assert MODULE.phone_class("T") == "stop"
    assert MODULE.phone_class("ZZZ") == "other"


def test_flatten_words_preserves_auditable_timing():
    result = {
        "segments": [{"words": [{"word": " Test", "start": 1.0, "end": 1.4, "probability": 0.9}]}]
    }
    phones = MODULE.flatten_words(result, {"test": [["T", "EH1", "S", "T"]]})
    assert [phone["phone"] for phone in phones] == ["T", "EH", "S", "T"]
    assert phones[0]["start_s"] == 1.0
    assert phones[-1]["end_s"] == 1.4


def test_aggregation_stays_clip_level():
    rows = []
    for event_index in [0, 1]:
        row = {"clip_id": "clip", "event_index": event_index}
        for column in MODULE.GEOMETRY:
            row[f"residual_{column}"] = 0.1 + event_index
            row[f"{column}_before"] = 0.0
            row[f"{column}_during"] = 0.2
            row[f"{column}_after"] = 0.1
        rows.append(row)
    aggregated = MODULE.aggregate_events(pd.DataFrame(rows))
    assert len(aggregated) == 1
    assert aggregated.loc[0, "event_count"] == 2
    assert aggregated.loc[0, "residual_inner_aperture_smooth_mean"] == pytest.approx(0.6)
