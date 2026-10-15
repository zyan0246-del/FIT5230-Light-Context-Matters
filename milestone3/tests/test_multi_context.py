import importlib.util
import json
from pathlib import Path

import numpy as np
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


@pytest.mark.parametrize("confidence, expected_threshold", [(0.9, 0.8), (0.7, 0.6)])
def test_clip_events_uses_recorded_confidence_fallback(tmp_path, monkeypatch, confidence, expected_threshold):
    monkeypatch.setattr(MODULE, "ROOT", tmp_path)
    asr = tmp_path / "outputs/m3_work/asr"
    features = tmp_path / "outputs/m3_work/pilot_features/features"
    asr.mkdir(parents=True)
    features.mkdir(parents=True)
    (asr / "sample.json").write_text(json.dumps({
        "segments": [{"words": [{
            "word": " bat", "start": 0.6, "end": 1.5, "probability": confidence,
        }]}],
    }), encoding="utf-8")
    times = np.arange(51) / 25
    frame = pd.DataFrame({"time_s": times, "detected": True})
    for column in MODULE.GEOMETRY:
        frame[column] = 1 + np.sin(times)
    frame.to_csv(features / "sample.csv", index=False)

    events = MODULE.clip_events("sample", {"bat": [["B", "AE1", "T"]]})
    assert len(events) == 1
    assert events[0]["selection_threshold"] == expected_threshold
