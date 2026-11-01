"""Reproducible FIT5230 M4 feasibility demonstrations for Yan Zhiheng.

The script deliberately uses only the public, already extracted M3 Strategy A
features.  It produces two small, auditable demonstrations:

1. Alignment-confidence-aware aggregation.  Mouth-transition events are
   aggregated either uniformly or with squared ASR word confidence.  This is a
   mechanism feasibility check, not forced alignment and not a new detector
   claim.
2. A bounded white-box feature attack.  Only the 12 dynamic summary features
   are moved toward a real-training centroid, with each change capped at 0.75
   training standard deviations.  This is a feature-space role-reversal PoC;
   it does not demonstrate that the perturbation can yet be rendered into an
   imperceptible MP4.

Every fold fits scaling, class prototypes and the classifier on training data
only.  Test labels are used solely for evaluation.
"""
from __future__ import annotations

import json
import hashlib
import platform
from pathlib import Path
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


SEED = 5230
REPEATS = 20
FOLDS = 4
CLASSIFIER_C = 0.1
ATTACK_ALPHA = 0.75
ATTACK_EPSILON_SD = 0.75
ATTACK_EPSILONS_SD = (0.0, 0.25, 0.50, 0.75, 1.00)
BOOTSTRAP_SAMPLES = 10_000
SIGNALS = (
    "inner_aperture_smooth",
    "outer_aperture_smooth",
    "lip_width_smooth",
)


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def demo_data_root() -> Path:
    return project_root() / "demo_data"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def m3_root() -> Path:
    candidate = project_root().parent / "FIT5230_M3_Repo" / "milestone3"
    if not candidate.exists():
        raise FileNotFoundError(
            "Expected the local M3 repository at "
            f"{candidate}. Set up FIT5230_M3_Repo beside this M4 folder."
        )
    return candidate


def load_public_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the distributable demo tables, with the local M3 tree as fallback."""
    demo = demo_data_root()
    manifest_path = demo / "manifest.csv"
    events_path = demo / "multi_context_events.csv"
    global_path = demo / "global_features.csv"
    if all(path.exists() for path in (manifest_path, events_path, global_path)):
        return (
            pd.read_csv(manifest_path),
            pd.read_csv(events_path),
            pd.read_csv(global_path),
        )

    m3 = m3_root()
    manifest = pd.read_csv(m3 / "data/m3_pilot_aligned_manifest.csv")
    events = pd.read_csv(
        m3 / "outputs/m3_work/multi_context_followup/multi_context_events.csv"
    )
    return manifest, events, global_features(m3, manifest)


def aggregate_events(events: pd.DataFrame, weighted: bool) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for clip_id, group in events.groupby("clip_id", sort=True):
        confidence = group.word_probability.to_numpy(float)
        weights = confidence**2 if weighted else np.ones(len(group), dtype=float)
        weights = weights / weights.sum()
        record: dict[str, float | str] = {
            "clip_id": clip_id,
            "event_count": int(len(group)),
            "mean_word_confidence": float(confidence.mean()),
        }
        for signal in SIGNALS:
            before = group[f"{signal}_before"].to_numpy(float)
            during = group[f"{signal}_during"].to_numpy(float)
            after = group[f"{signal}_after"].to_numpy(float)
            transitions = {
                "incoming": during - before,
                "outgoing": after - during,
                "curvature": after - 2 * during + before,
            }
            for name, values in transitions.items():
                record[f"{signal}_{name}_abs"] = float(
                    np.sum(weights * np.abs(values))
                )
        rows.append(record)
    return pd.DataFrame(rows)


def evaluate_representation(
    table: pd.DataFrame, feature_columns: list[str]
) -> pd.DataFrame:
    labels = table.label.map({"real": 0, "fake": 1}).to_numpy(int)
    matrix = table[feature_columns].to_numpy(float)
    rows: list[dict[str, float | int]] = []
    for repeat in range(REPEATS):
        probabilities = np.full(len(table), np.nan)
        splitter = StratifiedKFold(
            FOLDS, shuffle=True, random_state=SEED + repeat
        )
        for train, test in splitter.split(matrix, labels):
            scaler = StandardScaler().fit(matrix[train])
            model = LogisticRegression(
                C=CLASSIFIER_C,
                class_weight="balanced",
                max_iter=2000,
                random_state=SEED,
            ).fit(scaler.transform(matrix[train]), labels[train])
            probabilities[test] = model.predict_proba(
                scaler.transform(matrix[test])
            )[:, 1]
        rows.append(
            {
                "repeat": repeat,
                "macro_f1": float(
                    f1_score(labels, probabilities >= 0.5, average="macro")
                ),
                "auroc": float(roc_auc_score(labels, probabilities)),
            }
        )
    return pd.DataFrame(rows)


def global_features(m3: Path, manifest: pd.DataFrame) -> pd.DataFrame:
    sys.path.insert(0, str(m3 / "src"))
    from s2f.features import stats  # pylint: disable=import-error,import-outside-toplevel

    rows = []
    feature_dir = m3 / "outputs/m3_work/pilot_features/features"
    for clip_id in manifest.clip_id:
        frame = pd.read_csv(feature_dir / f"{clip_id}.csv")
        rows.append({"clip_id": clip_id, **stats(frame)})
    return pd.DataFrame(rows).merge(
        manifest[["clip_id", "label"]], on="clip_id", validate="one_to_one"
    )


def paired_bootstrap_summary(scores: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    """Summarise paired repeat-level changes without treating folds as independent."""
    summary: dict[str, dict[str, float | int]] = {}
    for offset, metric in enumerate(("macro_f1", "auroc")):
        pivot = scores.pivot(index="repeat", columns="model", values=metric)
        differences = (
            pivot["confidence_squared"] - pivot["uniform_events"]
        ).to_numpy(float)
        rng = np.random.default_rng(SEED + 100 + offset)
        bootstrapped_means = rng.choice(
            differences,
            size=(BOOTSTRAP_SAMPLES, len(differences)),
            replace=True,
        ).mean(axis=1)
        summary[metric] = {
            "mean_paired_difference": float(differences.mean()),
            "bootstrap_95_ci_low": float(np.quantile(bootstrapped_means, 0.025)),
            "bootstrap_95_ci_high": float(np.quantile(bootstrapped_means, 0.975)),
            "confidence_weighted_wins": int(np.sum(differences > 0)),
            "ties": int(np.sum(differences == 0)),
            "repeats": int(len(differences)),
        }
    return summary


def bounded_role_reversal_attack(
    global_table: pd.DataFrame, epsilon_sd: float
) -> pd.DataFrame:
    feature_columns = [
        column
        for column in global_table.columns
        if column not in {"clip_id", "label"}
    ]
    dynamic_indices = np.array(
        [
            index
            for index, column in enumerate(feature_columns)
            if column.startswith(
                ("aperture_velocity", "aperture_acceleration", "aperture_jerk")
            )
        ],
        dtype=int,
    )
    if len(dynamic_indices) != 12:
        raise AssertionError(
            f"Expected 12 dynamic summaries, found {len(dynamic_indices)}"
        )

    matrix = global_table[feature_columns].to_numpy(float)
    labels = global_table.label.map({"real": 0, "fake": 1}).to_numpy(int)
    rows: list[dict[str, float | int | str]] = []
    for repeat in range(REPEATS):
        splitter = StratifiedKFold(
            FOLDS, shuffle=True, random_state=SEED + repeat
        )
        for fold, (train, test) in enumerate(splitter.split(matrix, labels)):
            scaler = StandardScaler().fit(matrix[train])
            train_z = scaler.transform(matrix[train])
            test_z = scaler.transform(matrix[test])
            model = LogisticRegression(
                C=CLASSIFIER_C,
                class_weight="balanced",
                max_iter=2000,
                random_state=SEED,
            ).fit(train_z, labels[train])
            clean_probability = model.predict_proba(test_z)[:, 1]
            real_centroid = train_z[labels[train] == 0].mean(axis=0)
            attacked_z = test_z.copy()
            for local_index, global_index in enumerate(test):
                if labels[global_index] != 1:
                    continue
                desired = ATTACK_ALPHA * (real_centroid - attacked_z[local_index])
                bounded = np.clip(
                    desired, -epsilon_sd, epsilon_sd
                )
                attacked_z[local_index, dynamic_indices] += bounded[dynamic_indices]
            attacked_probability = model.predict_proba(attacked_z)[:, 1]

            for local_index, global_index in enumerate(test):
                if labels[global_index] != 1:
                    continue
                rows.append(
                    {
                        "repeat": repeat,
                        "fold": fold,
                        "epsilon_training_sd": epsilon_sd,
                        "clip_id": global_table.iloc[global_index].clip_id,
                        "clean_fake_probability": float(
                            clean_probability[local_index]
                        ),
                        "attacked_fake_probability": float(
                            attacked_probability[local_index]
                        ),
                        "probability_drop": float(
                            clean_probability[local_index]
                            - attacked_probability[local_index]
                        ),
                        "clean_evades": int(clean_probability[local_index] < 0.5),
                        "attacked_evades": int(
                            attacked_probability[local_index] < 0.5
                        ),
                    }
                )
    return pd.DataFrame(rows)


def attack_repeat_bootstrap_summary(
    attack_sweep: pd.DataFrame,
) -> pd.DataFrame:
    """Quantify attack stability over repeat seeds, not repeated clip rows."""
    rows: list[dict[str, float | int]] = []
    for offset, (epsilon, frame) in enumerate(
        attack_sweep.groupby("epsilon_training_sd", sort=True)
    ):
        by_repeat = frame.groupby("repeat", as_index=False).agg(
            probability_drop=("probability_drop", "mean"),
            evasion_change=(
                "attacked_evades",
                lambda values: float(values.mean()),
            ),
            clean_evasion=("clean_evades", "mean"),
        )
        by_repeat["evasion_change"] -= by_repeat["clean_evasion"]
        rng = np.random.default_rng(SEED + 300 + offset)
        record: dict[str, float | int] = {
            "epsilon_training_sd": float(epsilon),
            "repeats": int(len(by_repeat)),
        }
        for metric in ("probability_drop", "evasion_change"):
            values = by_repeat[metric].to_numpy(float)
            boot = rng.choice(
                values,
                size=(BOOTSTRAP_SAMPLES, len(values)),
                replace=True,
            ).mean(axis=1)
            record[f"{metric}_mean"] = float(values.mean())
            record[f"{metric}_ci_low"] = float(np.quantile(boot, 0.025))
            record[f"{metric}_ci_high"] = float(np.quantile(boot, 0.975))
            record[f"{metric}_positive_repeats"] = int(np.sum(values > 0))
        rows.append(record)
    return pd.DataFrame(rows)


def main() -> None:
    root = project_root()
    out = root / "results"
    figures = root / "figures"
    out.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    manifest, events, global_table = load_public_inputs()
    event_columns = [
        f"{signal}_{transition}_abs"
        for signal in SIGNALS
        for transition in ("incoming", "outgoing", "curvature")
    ]
    enhancement_rows = []
    enhancement_repeats = []
    for mode, weighted in (
        ("uniform_events", False),
        ("confidence_squared", True),
    ):
        aggregated = aggregate_events(events, weighted).merge(
            manifest[["clip_id", "label"]],
            on="clip_id",
            validate="one_to_one",
        )
        scores = evaluate_representation(aggregated, event_columns)
        scores.insert(1, "model", mode)
        enhancement_repeats.append(scores)
        enhancement_rows.append(
            {
                "model": mode,
                "n_clips": len(aggregated),
                "n_events": len(events),
                "macro_f1_mean": scores.macro_f1.mean(),
                "macro_f1_sd": scores.macro_f1.std(),
                "auroc_mean": scores.auroc.mean(),
                "auroc_sd": scores.auroc.std(),
            }
        )
    enhancement = pd.DataFrame(enhancement_rows)
    enhancement_repeats_frame = pd.concat(
        enhancement_repeats, ignore_index=True
    )
    enhancement.to_csv(out / "enhancement_summary.csv", index=False)
    enhancement_repeats_frame.to_csv(
        out / "enhancement_repeats.csv", index=False
    )
    enhancement_paired = paired_bootstrap_summary(enhancement_repeats_frame)
    (out / "enhancement_paired_summary.json").write_text(
        json.dumps(enhancement_paired, indent=2), encoding="utf-8"
    )

    attack_sweep = pd.concat(
        [
            bounded_role_reversal_attack(global_table, epsilon)
            for epsilon in ATTACK_EPSILONS_SD
        ],
        ignore_index=True,
    )
    attack_sweep.to_csv(
        out / "role_reversal_attack_sweep_predictions.csv", index=False
    )
    attack_sweep_summary = (
        attack_sweep.groupby("epsilon_training_sd", as_index=False)
        .agg(
            n_fake_evaluations=("clip_id", "size"),
            mean_clean_fake_probability=("clean_fake_probability", "mean"),
            mean_attacked_fake_probability=("attacked_fake_probability", "mean"),
            mean_probability_drop=("probability_drop", "mean"),
            clean_evasion_fraction=("clean_evades", "mean"),
            attacked_evasion_fraction=("attacked_evades", "mean"),
        )
    )
    attack_sweep_summary.to_csv(
        out / "role_reversal_attack_sweep_summary.csv", index=False
    )
    attack_stability = attack_repeat_bootstrap_summary(attack_sweep)
    attack_stability.to_csv(
        out / "role_reversal_attack_stability.csv", index=False
    )
    attack = attack_sweep.loc[
        attack_sweep.epsilon_training_sd.eq(ATTACK_EPSILON_SD)
    ].copy()
    attack.to_csv(out / "role_reversal_attack_predictions.csv", index=False)
    attack_summary = {
        "n_public_clips": int(len(global_table)),
        "n_fake_evaluations": int(len(attack)),
        "repeats": REPEATS,
        "folds": FOLDS,
        "dynamic_features_modified": 12,
        "epsilon_training_sd": ATTACK_EPSILON_SD,
        "alpha_toward_real_training_centroid": ATTACK_ALPHA,
        "mean_clean_fake_probability": float(
            attack.clean_fake_probability.mean()
        ),
        "mean_attacked_fake_probability": float(
            attack.attacked_fake_probability.mean()
        ),
        "mean_probability_drop": float(attack.probability_drop.mean()),
        "clean_evasion_fraction": float(attack.clean_evades.mean()),
        "attacked_evasion_fraction": float(attack.attacked_evades.mean()),
        "repeat_level_stability": attack_stability.loc[
            attack_stability.epsilon_training_sd.eq(ATTACK_EPSILON_SD)
        ].iloc[0].to_dict(),
        "claim_boundary": (
            "Feature-space feasibility only. The bounded dynamic-summary "
            "changes have not been rendered into an imperceptible MP4."
        ),
    }
    (out / "role_reversal_attack_summary.json").write_text(
        json.dumps(attack_summary, indent=2), encoding="utf-8"
    )

    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    models = ["uniform_events", "confidence_squared"]
    labels_plot = ["Uniform", "Confidence\nweighted"]
    for x_value, metric, ylabel in (
        (0, "macro_f1", "Macro-F1"),
        (1, "auroc", "AUROC"),
    ):
        data = [
            enhancement_repeats_frame.loc[
                enhancement_repeats_frame.model.eq(model), metric
            ]
            for model in models
        ]
        axes[x_value].boxplot(data, tick_labels=labels_plot)
        axes[x_value].set_ylim(0.35, 0.8)
        axes[x_value].set_ylabel(ylabel)
        axes[x_value].grid(axis="y", alpha=0.25)
    figure.suptitle("PoC 1: confidence-aware event aggregation (24 clips)")
    figure.tight_layout()
    figure.savefig(figures / "enhancement_poc.png", dpi=200)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    axes[0].scatter(
        attack.clean_fake_probability,
        attack.attacked_fake_probability,
        s=18,
        alpha=0.45,
        color="#1769aa",
    )
    axes[0].plot([0, 1], [0, 1], "--", color="0.45", linewidth=1)
    axes[0].axhline(0.5, color="#c62828", linestyle=":", linewidth=1)
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    axes[0].set_xlabel("Clean fake probability")
    axes[0].set_ylabel("Attacked fake probability")
    axes[0].set_title("Individual predictions at 0.75 SD")
    axes[0].grid(alpha=0.2)

    axes[1].plot(
        attack_sweep_summary.epsilon_training_sd,
        attack_sweep_summary.mean_attacked_fake_probability,
        marker="o",
        color="#1769aa",
        label="Mean fake probability",
    )
    axes[1].plot(
        attack_sweep_summary.epsilon_training_sd,
        attack_sweep_summary.attacked_evasion_fraction,
        marker="s",
        color="#c62828",
        label="Evasion fraction",
    )
    axes[1].set_ylim(0.40, 0.61)
    axes[1].set_xlabel("Per-feature attack cap (training SD)")
    axes[1].set_title("Attack-budget sweep")
    axes[1].grid(alpha=0.2)
    axes[1].legend(fontsize=8)
    figure.suptitle("PoC 2: bounded dynamic-feature role reversal")
    figure.tight_layout()
    figure.savefig(figures / "role_reversal_poc.png", dpi=200)
    plt.close(figure)

    run_record = {
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "input_sha256": {
            name: sha256_file(demo_data_root() / name)
            for name in (
                "manifest.csv",
                "multi_context_events.csv",
                "global_features.csv",
            )
            if (demo_data_root() / name).exists()
        },
        "seed": SEED,
        "repeats": REPEATS,
        "folds": FOLDS,
        "classifier": "standardised class-balanced logistic regression",
        "classifier_C": CLASSIFIER_C,
        "enhancement": enhancement.to_dict("records"),
        "enhancement_paired": enhancement_paired,
        "attack": attack_summary,
        "attack_sweep": attack_sweep_summary.to_dict("records"),
        "attack_repeat_stability": attack_stability.to_dict("records"),
        "data_boundary": (
            "24 public M3 pilot clips; one clip per source/speaker ID in this "
            "branch; results are repeated-split feasibility evidence only."
        ),
        "claim_checks": {
            "enhancement_significant_improvement": False,
            "attack_score_response_stable": True,
            "attack_threshold_crossing_stable": False,
            "rendered_adversarial_video_completed": False,
        },
    }
    (out / "run_record.json").write_text(
        json.dumps(run_record, indent=2), encoding="utf-8"
    )

    print(enhancement.to_string(index=False))
    print(json.dumps(attack_summary, indent=2))


if __name__ == "__main__":
    main()
