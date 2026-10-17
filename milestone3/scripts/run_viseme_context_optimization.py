"""Low-dimensional viseme-context follow-up for the matched course pilot.

This experiment addresses the two largest problems in the original weak-phone
representation: hundreds of features for only 18 outer-fold training clips,
and phone labels that are more detailed than the visible mouth configuration.

The comparison is deliberately controlled.  The low-dimensional window,
viseme-context and shuffled-viseme models use the same numeric trajectories,
outer source-pair folds, inner source-pair tuning folds, classifier family and
0.5 decision threshold.  Viseme labels are derived from the already audited
ARPAbet proxy and therefore do not upgrade the alignment to ground truth.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from s2f.features import build_features, read_intervals
from s2f.manifest import validate_manifest


ROOT = Path(__file__).resolve().parents[1]
SEED = 5230
REPEATS = 20
OUTER_FOLDS = 4
INNER_FOLDS = 3
THRESHOLD = 0.5
C_GRID = (0.001, 0.01, 0.1, 1.0)

# A deliberately coarse, auditable ARPAbet-to-visible-articulation mapping.
# It reduces sparsity; it is not claimed as frame-accurate viseme ground truth.
VISEME_GROUPS = {
    "v_open": {"AA", "AE", "AH", "AW", "AY"},
    "v_round": {"AO", "OW", "OY", "UH", "UW"},
    "v_front": {"EH", "ER", "EY", "IH", "IY"},
    "bilabial_closed": {"P", "B", "M"},
    "labiodental": {"F", "V"},
    "dental": {"TH", "DH"},
    "alveolar": {"T", "D", "S", "Z", "N", "L"},
    "postalveolar": {"SH", "ZH", "CH", "JH", "R"},
    "velar": {"K", "G", "NG"},
    "rounded_glide": {"W"},
    "palatal_glide": {"Y"},
    "glottal": {"HH"},
}

GEOMETRY = ("inner_aperture_smooth", "outer_aperture_smooth", "lip_width_smooth")
PHASES = ("before", "during", "after")


def clean_phone(phone: str) -> str:
    return re.sub(r"\d", "", str(phone).upper())


def phone_to_viseme(phone: str) -> str:
    phone = clean_phone(phone)
    for name, phones in VISEME_GROUPS.items():
        if phone in phones:
            return name
    return "other"


def grouped_folds(groups: pd.Series, folds: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    unique = np.array(sorted(groups.astype(str).unique()))
    if len(unique) < folds:
        raise ValueError(f"Need at least {folds} source pairs, received {len(unique)}")
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    result = []
    for test_groups in np.array_split(unique, folds):
        test = groups.astype(str).isin(test_groups).to_numpy()
        result.append((np.flatnonzero(~test), np.flatnonzero(test)))
    return result


def lowdim_numeric(row: dict) -> dict[str, float]:
    """Select interpretable trajectory summaries instead of all 102 windows."""
    result: dict[str, float] = {}
    for signal in GEOMETRY:
        phase_means = {}
        for phase in PHASES:
            for statistic in ("mean", "std"):
                source = f"ctx_{phase}_{signal}_{statistic}"
                result[source] = float(row[source])
            phase_means[phase] = float(row[f"ctx_{phase}_{signal}_mean"])
        result[f"transition_in_{signal}"] = phase_means["during"] - phase_means["before"]
        result[f"transition_out_{signal}"] = phase_means["after"] - phase_means["during"]
        result[f"transition_total_{signal}"] = phase_means["after"] - phase_means["before"]

    # Velocity describes direction and smoothness without unstable acceleration/jerk.
    signal = "aperture_velocity"
    for phase in PHASES:
        for statistic in ("mean", "std", "abs_mean"):
            source = f"ctx_{phase}_{signal}_{statistic}"
            result[source] = float(row[source])
    return result


def make_global_records(features: pd.DataFrame) -> list[dict]:
    """Keep only the original 24 global numeric summaries."""
    return [global_numeric(row) for row in features.to_dict("records")]


def global_numeric(row: dict) -> dict[str, float]:
    return {
        key: float(value)
        for key, value in row.items()
        if key != "clip_id" and not key.startswith("ctx_")
        and not key.endswith("_arpabet") and not key.endswith("_viseme")
        and key not in {"previous_phoneme", "central_phoneme", "next_phoneme"}
    }


def _add_interaction(result: dict[str, float], name: str, value: float, category: str) -> None:
    result[f"interaction={name}|{category}"] = float(value)


def make_lowdim_records(features: pd.DataFrame, *, use_viseme: bool) -> list[dict]:
    rows: list[dict] = []
    for row in features.to_dict("records"):
        # Retain the proven global baseline and add a compact local trajectory.
        # The first low-dimensional attempt omitted the global block and was
        # retained only as an internal diagnostic, not as evidence of benefit.
        result = {**global_numeric(row), **lowdim_numeric(row)}
        if use_viseme:
            previous = str(row["previous_viseme"])
            central = str(row["central_viseme"])
            following = str(row["next_viseme"])
            result.update({
                f"previous_viseme={previous}": 1.0,
                f"central_viseme={central}": 1.0,
                f"next_viseme={following}": 1.0,
            })
            # Only a small, hypothesis-led set of interactions is used.  The
            # incoming transition is conditioned on the previous viseme and
            # the outgoing transition on the following viseme.
            for signal in GEOMETRY:
                before = float(row[f"ctx_before_{signal}_mean"])
                during = float(row[f"ctx_during_{signal}_mean"])
                after = float(row[f"ctx_after_{signal}_mean"])
                _add_interaction(result, f"incoming_{signal}", during - before, previous)
                _add_interaction(result, f"outgoing_{signal}", after - during, following)
            _add_interaction(
                result,
                "central_aperture",
                float(row["ctx_during_inner_aperture_smooth_mean"]),
                central,
            )
            _add_interaction(
                result,
                "central_width",
                float(row["ctx_during_lip_width_smooth_mean"]),
                central,
            )
        rows.append(result)
    return rows


def attach_visemes(features: pd.DataFrame, intervals: pd.DataFrame) -> pd.DataFrame:
    exact = intervals.reset_index()[[
        "clip_id", "previous_phone_arpabet", "central_phone_arpabet", "next_phone_arpabet",
    ]]
    result = features.merge(exact, on="clip_id", validate="one_to_one")
    for position in ("previous", "central", "next"):
        result[f"{position}_viseme"] = result[f"{position}_phone_arpabet"].map(phone_to_viseme)
    return result


def shuffled_visemes(features: pd.DataFrame, manifest: pd.DataFrame, seed: int) -> pd.DataFrame:
    result = features.copy()
    columns = ["previous_viseme", "central_viseme", "next_viseme"]
    paired = manifest[["source_id", "clip_id"]].merge(
        features[["clip_id"] + columns], on="clip_id", validate="one_to_one",
    )
    if (paired.groupby("source_id")[columns].nunique() > 1).any().any():
        raise ValueError("Matched real/fake clips must share the same viseme labels")
    unique = paired.groupby("source_id", sort=True)[columns].first().reset_index()
    rng = np.random.default_rng(seed)
    permuted = unique.copy()
    permuted[columns] = unique.loc[rng.permutation(len(unique)), columns].to_numpy()
    mapping = permuted.set_index("source_id")
    clip_to_source = manifest.set_index("clip_id").source_id
    for column in columns:
        result[column] = result.clip_id.map(clip_to_source).map(mapping[column])
    return result


def classifier(c_value: float) -> Pipeline:
    return Pipeline([
        ("vectorize", DictVectorizer(sparse=False)),
        ("scale", StandardScaler()),
        ("classifier", LogisticRegression(
            C=c_value, class_weight="balanced", max_iter=4000, random_state=SEED,
        )),
    ])


def score(labels: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    prediction = probability >= THRESHOLD
    return {
        "macro_f1": float(f1_score(labels, prediction, average="macro", zero_division=0)),
        "auroc": float(roc_auc_score(labels, probability)),
        "log_loss": float(log_loss(labels, probability, labels=[0, 1])),
    }


def select_c(
    model_records: list[dict], labels: np.ndarray, groups: pd.Series, train_index: np.ndarray,
    *, seed: int,
) -> tuple[float, list[dict]]:
    """Tune regularisation only inside an outer training fold using log-loss."""
    local_groups = groups.iloc[train_index].reset_index(drop=True)
    local_labels = labels[train_index]
    local_records = [model_records[index] for index in train_index]
    rows = []
    for c_value in C_GRID:
        probabilities = np.full(len(train_index), np.nan)
        for inner_train, inner_test in grouped_folds(local_groups, INNER_FOLDS, seed):
            model = classifier(c_value)
            model.fit([local_records[index] for index in inner_train], local_labels[inner_train])
            probabilities[inner_test] = model.predict_proba(
                [local_records[index] for index in inner_test]
            )[:, 1]
        metrics = score(local_labels, probabilities)
        rows.append({"C": c_value, **metrics})
    # Log-loss is threshold independent and gives a deterministic tie break
    # towards stronger regularisation (smaller C).
    selected = min(rows, key=lambda row: (row["log_loss"], row["C"]))["C"]
    return float(selected), rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/matched_sadtalker_aligned_manifest.csv")
    parser.add_argument("--intervals", default="data/matched_sadtalker_intervals.csv")
    parser.add_argument("--features", default="outputs/matched_sadtalker/features")
    parser.add_argument("--out", default="artifacts/viseme_optimization")
    parser.add_argument(
        "--new-segments-only", action="store_true",
        help="Evaluate only the non-overlapping s1/s2 segments in the expanded pilot",
    )
    args = parser.parse_args()

    manifest = validate_manifest(ROOT / args.manifest, ROOT, protocol="speaker")
    if args.new_segments_only:
        manifest = manifest.loc[
            manifest.clip_id.str.contains(r"_s[12]$", regex=True)
        ].reset_index(drop=True)
        if len(manifest) != 48 or manifest.source_id.nunique() != 24:
            raise ValueError("Expected 24 complete new-segment pairs / 48 clips")
    intervals_frame = pd.read_csv(ROOT / args.intervals, keep_default_na=False)
    required_exact = {
        "previous_phone_arpabet", "central_phone_arpabet", "next_phone_arpabet",
    }
    if required_exact - set(intervals_frame.columns):
        raise ValueError("Intervals are missing auditable exact ARPAbet columns")
    intervals = read_intervals(ROOT / args.intervals)
    features = build_features(manifest, ROOT / args.features, intervals=intervals)
    features = attach_visemes(features, intervals_frame.set_index("clip_id"))
    features = manifest[["clip_id"]].merge(features, on="clip_id", validate="one_to_one")
    labels = manifest.label.map({"real": 0, "fake": 1}).to_numpy(int)
    # Speaker grouping is identical to pair grouping in the original 12-pair
    # pilot and remains leakage-safe when multiple segments per speaker exist.
    groups = manifest.speaker_id

    modes = ("global", "lowdim_window", "viseme_context", "shuffled_viseme")
    fold_rows: list[dict] = []
    repeat_rows: list[dict] = []
    prediction_rows: list[dict] = []
    tuning_rows: list[dict] = []

    for repeat in range(REPEATS):
        shuffled = shuffled_visemes(features, manifest, SEED + repeat)
        mode_records = {
            "global": make_global_records(features),
            "lowdim_window": make_lowdim_records(features, use_viseme=False),
            "viseme_context": make_lowdim_records(features, use_viseme=True),
            "shuffled_viseme": make_lowdim_records(shuffled, use_viseme=True),
        }
        probabilities = {mode: np.full(len(manifest), np.nan) for mode in modes}
        for fold, (train_index, test_index) in enumerate(
            grouped_folds(groups, OUTER_FOLDS, SEED + repeat)
        ):
            if set(groups.iloc[train_index]) & set(groups.iloc[test_index]):
                raise AssertionError("Source-pair leakage")
            for mode in modes:
                selected_c, tuning = select_c(
                    mode_records[mode], labels, groups, train_index,
                    seed=SEED + repeat * 100 + fold,
                )
                for row in tuning:
                    tuning_rows.append({
                        "repeat": repeat, "fold": fold, "model": mode,
                        "selected": row["C"] == selected_c, **row,
                    })
                model = classifier(selected_c)
                model.fit(
                    [mode_records[mode][index] for index in train_index], labels[train_index]
                )
                probability = model.predict_proba(
                    [mode_records[mode][index] for index in test_index]
                )[:, 1]
                probabilities[mode][test_index] = probability
                fold_rows.append({
                    "repeat": repeat, "fold": fold, "model": mode,
                    "selected_C": selected_c,
                    "n_train": int(len(train_index)), "n_test": int(len(test_index)),
                    "n_features": int(len(model.named_steps["vectorize"].get_feature_names_out())),
                    **score(labels[test_index], probability),
                })
        for mode in modes:
            metrics = score(labels, probabilities[mode])
            repeat_rows.append({"repeat": repeat, "model": mode, "n": len(labels), **metrics})
            for index, probability in enumerate(probabilities[mode]):
                prediction_rows.append({
                    "repeat": repeat, "model": mode,
                    "clip_id": manifest.iloc[index].clip_id,
                    "source_id": manifest.iloc[index].source_id,
                    "label": manifest.iloc[index].label,
                    "fake_probability": float(probability),
                    "prediction": int(probability >= THRESHOLD),
                })

    folds = pd.DataFrame(fold_rows)
    repeats = pd.DataFrame(repeat_rows)
    predictions = pd.DataFrame(prediction_rows)
    tuning = pd.DataFrame(tuning_rows)
    summary = repeats.groupby("model").agg(
        macro_f1_mean=("macro_f1", "mean"), macro_f1_sd=("macro_f1", "std"),
        auroc_mean=("auroc", "mean"), auroc_sd=("auroc", "std"),
        log_loss_mean=("log_loss", "mean"), log_loss_sd=("log_loss", "std"),
    ).reset_index()

    pivot = repeats.pivot(index="repeat", columns="model", values=["macro_f1", "auroc", "log_loss"])
    comparison_pairs = [
        ("lowdim_window", "global"),
        ("viseme_context", "lowdim_window"),
        ("viseme_context", "shuffled_viseme"),
    ]
    comparison_rows = []
    for treatment, control in comparison_pairs:
        for metric in ("macro_f1", "auroc", "log_loss"):
            delta = pivot[(metric, treatment)] - pivot[(metric, control)]
            comparison_rows.append({
                "comparison": f"{treatment}-minus-{control}", "metric": metric,
                "mean_delta": float(delta.mean()), "sd_delta": float(delta.std()),
                "positive_repeats": int((delta > 0).sum()),
                "ties": int((delta == 0).sum()),
                "negative_repeats": int((delta < 0).sum()),
            })
    comparisons = pd.DataFrame(comparison_rows)

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    folds.to_csv(out / "fold_metrics.csv", index=False)
    repeats.to_csv(out / "repeat_oof_metrics.csv", index=False)
    predictions.to_csv(out / "oof_predictions.csv", index=False)
    tuning.to_csv(out / "inner_tuning.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    comparisons.to_csv(out / "paired_comparisons.csv", index=False)
    pair_labels = manifest[["source_id", "clip_id"]].merge(
        features[["clip_id", "previous_viseme", "central_viseme", "next_viseme"]],
        on="clip_id", validate="one_to_one",
    )
    pair_labels.drop(columns="clip_id").drop_duplicates().to_csv(
        out / "pair_viseme_labels.csv", index=False,
    )

    feature_ranges = folds.groupby("model").n_features.agg(["min", "max"]).to_dict("index")
    selected_c_counts = (
        folds.groupby(["model", "selected_C"]).size().rename("count").reset_index()
        .to_dict("records")
    )
    (out / "run.json").write_text(json.dumps({
        "seed": SEED, "repeats": REPEATS, "outer_folds": OUTER_FOLDS,
        "inner_folds": INNER_FOLDS, "threshold": THRESHOLD,
        "C_grid": list(C_GRID), "C_selection": "minimum inner grouped-CV log-loss",
        "group": "speaker_id (all matched segments from a speaker stay together)",
        "evaluation_subset": "new_non_overlapping_segments_s1_s2" if args.new_segments_only else "all_manifest_segments",
        "feature_count_range_by_model": feature_ranges,
        "selected_C_counts": selected_c_counts,
        "alignment_boundary": (
            "Visemes come from Whisper tiny.en word timestamps plus uniform CMUdict phone placement; "
            "this remains a weak proxy, not forced alignment."
        ),
        "claim_boundary": (
            "Small 12-pair course pilot. Repeated folds quantify split sensitivity on the same pairs; "
            "they are not independent datasets."
        ),
    }, indent=2), encoding="utf-8")

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    labels_plot = ["Global", "Low-dim\nwindow", "Viseme\ncontext", "Shuffled\nviseme"]
    for axis, metric, title in [
        (axes[0], "macro_f1", "Out-of-fold Macro-F1"),
        (axes[1], "auroc", "Out-of-fold AUROC"),
    ]:
        data = [repeats.loc[repeats.model.eq(mode), metric] for mode in modes]
        axis.boxplot(data, tick_labels=labels_plot)
        axis.set_ylim(0, 1.02)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    subset_title = "new non-overlapping segments" if args.new_segments_only else "all matched segments"
    figure.suptitle(f"Low-dimensional viseme follow-up: {subset_title}")
    figure.tight_layout()
    figure.savefig(out / "viseme_optimization.png", dpi=180)
    plt.close(figure)

    print(summary.to_string(index=False))
    print(comparisons.to_string(index=False))
    print("outputs", out)


if __name__ == "__main__":
    main()
