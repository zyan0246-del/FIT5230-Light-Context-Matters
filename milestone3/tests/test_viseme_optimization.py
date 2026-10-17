import importlib.util
from pathlib import Path

import pandas as pd
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_viseme_context_optimization.py"
SPEC = importlib.util.spec_from_file_location("viseme_optimization", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_phone_to_viseme_is_coarse_and_stress_invariant():
    assert MODULE.phone_to_viseme("P") == "bilabial_closed"
    assert MODULE.phone_to_viseme("M") == "bilabial_closed"
    assert MODULE.phone_to_viseme("UW1") == "v_round"
    assert MODULE.phone_to_viseme("IY0") == "v_front"
    assert MODULE.phone_to_viseme("?") == "other"


def test_lowdim_numeric_has_fixed_small_dimension():
    row = {}
    for signal in MODULE.GEOMETRY:
        for phase_index, phase in enumerate(MODULE.PHASES):
            row[f"ctx_{phase}_{signal}_mean"] = 0.1 + phase_index
            row[f"ctx_{phase}_{signal}_std"] = 0.2
    for phase in MODULE.PHASES:
        row[f"ctx_{phase}_aperture_velocity_mean"] = 0.1
        row[f"ctx_{phase}_aperture_velocity_std"] = 0.2
        row[f"ctx_{phase}_aperture_velocity_abs_mean"] = 0.3
    result = MODULE.lowdim_numeric(row)
    assert len(result) == 36
    assert result["transition_in_inner_aperture_smooth"] == 1.0
    assert result["transition_out_inner_aperture_smooth"] == 1.0


def test_viseme_records_add_only_hypothesis_led_interactions():
    row = {"clip_id": "sample"}
    for index in range(24):
        row[f"global_{index}"] = index / 10
    for signal in MODULE.GEOMETRY:
        for phase_index, phase in enumerate(MODULE.PHASES):
            row[f"ctx_{phase}_{signal}_mean"] = 0.1 + phase_index
            row[f"ctx_{phase}_{signal}_std"] = 0.2
    for phase in MODULE.PHASES:
        row[f"ctx_{phase}_aperture_velocity_mean"] = 0.1
        row[f"ctx_{phase}_aperture_velocity_std"] = 0.2
        row[f"ctx_{phase}_aperture_velocity_abs_mean"] = 0.3
    row.update(previous_viseme="bilabial_closed", central_viseme="v_open", next_viseme="alveolar")
    record = MODULE.make_lowdim_records(pd.DataFrame([row]), use_viseme=True)[0]
    assert 60 < len(record) <= 75
    assert record["previous_viseme=bilabial_closed"] == 1.0
    assert "interaction=incoming_inner_aperture_smooth|bilabial_closed" in record


def test_grouped_folds_keep_pairs_together():
    groups = pd.Series([f"pair_{index}" for index in range(12) for _ in range(2)])
    folds = MODULE.grouped_folds(groups, folds=4, seed=5230)
    seen = []
    for train_index, test_index in folds:
        assert not set(groups.iloc[train_index]) & set(groups.iloc[test_index])
        seen.extend(test_index.tolist())
    assert sorted(seen) == list(range(24))


def test_expanded_manifest_has_complete_pairs_and_speaker_disjoint_splits():
    path = SCRIPT.parents[1] / "data" / "expanded_sadtalker_manifest.csv"
    if not path.exists():
        # The manifest is generated on the GPU server and committed with the
        # completed evidence.  Unit-only environments may not have run it yet.
        return
    manifest = pd.read_csv(path, keep_default_na=False)
    if len(manifest) != 72:
        pytest.skip("Expanded GPU generation is still in progress")
    assert len(manifest) == 72
    assert manifest.source_id.nunique() == 36
    assert manifest.speaker_id.nunique() == 12
    assert (manifest.groupby("source_id").size() == 2).all()
    assert all(set(group.label) == {"real", "fake"} for _, group in manifest.groupby("source_id"))
    assert (manifest.groupby("speaker_id").source_id.nunique() == 3).all()
    assert (manifest.groupby("speaker_id").split.nunique() == 1).all()


def test_speaker_grouped_folds_keep_all_three_segments_together():
    groups = pd.Series([f"speaker_{index}" for index in range(12) for _ in range(6)])
    folds = MODULE.grouped_folds(groups, folds=4, seed=5230)
    for train_index, test_index in folds:
        assert not set(groups.iloc[train_index]) & set(groups.iloc[test_index])
        assert len(test_index) == 18
