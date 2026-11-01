"""Independent consistency checks for the public FIT5230 M4 demo outputs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


record = json.loads((ROOT / "results/run_record.json").read_text())
stability = pd.read_csv(ROOT / "results/role_reversal_attack_stability.csv")

assert record["seed"] == 5230
assert record["repeats"] == 20 and record["folds"] == 4
assert {row["n_clips"] for row in record["enhancement"]} == {24}
assert {row["n_events"] for row in record["enhancement"]} == {547}
assert record["claim_checks"] == {
    "enhancement_significant_improvement": False,
    "attack_score_response_stable": True,
    "attack_threshold_crossing_stable": False,
    "rendered_adversarial_video_completed": False,
}

for filename, expected in record["input_sha256"].items():
    assert sha256_file(ROOT / "demo_data" / filename) == expected

selected = stability.loc[stability.epsilon_training_sd.eq(0.75)].iloc[0]
assert selected.probability_drop_ci_low > 0
assert selected.probability_drop_positive_repeats == 20
assert selected.evasion_change_ci_low <= 1e-12
assert selected.evasion_change_positive_repeats == 5

sweep = pd.DataFrame(record["attack_sweep"])
assert sweep.mean_attacked_fake_probability.is_monotonic_decreasing

print("M4 public demo verification: all checks passed.")
