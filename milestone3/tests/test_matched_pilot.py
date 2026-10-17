import importlib.util
from pathlib import Path

import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_matched_sadtalker_evaluation.py"
SPEC = importlib.util.spec_from_file_location("matched_evaluation", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_grouped_folds_keep_pairs_together_and_cover_every_clip():
    groups = pd.Series([f"pair_{index}" for index in range(12) for _ in range(2)])
    folds = MODULE.grouped_folds(groups, seed=5230)
    seen = []
    for train_index, test_index in folds:
        train_groups = set(groups.iloc[train_index])
        test_groups = set(groups.iloc[test_index])
        assert not train_groups & test_groups
        assert len(test_groups) == 3
        assert len(test_index) == 6
        seen.extend(test_index.tolist())
    assert sorted(seen) == list(range(24))


def test_phone_shuffle_preserves_pair_consistency_and_category_multiset():
    rows = []
    contexts = []
    for index in range(12):
        for label in ["real", "fake"]:
            clip_id = f"{label}_{index}"
            rows.append({"clip_id": clip_id, "source_id": f"pair_{index}"})
            contexts.append({
                "clip_id": clip_id,
                "previous_phoneme": f"previous_{index}",
                "central_phoneme": f"central_{index}",
                "next_phoneme": f"next_{index}",
            })
    manifest = pd.DataFrame(rows)
    features = pd.DataFrame(contexts)
    shuffled = MODULE.shuffled_phone_features(features, manifest, seed=5230)
    joined = manifest.merge(shuffled, on="clip_id")
    assert not (joined.groupby("source_id")[MODULE.CONTEXT].nunique() > 1).any().any()
    for column in MODULE.CONTEXT:
        before = sorted(features.drop_duplicates(column)[column])
        after = sorted(shuffled.drop_duplicates(column)[column])
        assert before == after
