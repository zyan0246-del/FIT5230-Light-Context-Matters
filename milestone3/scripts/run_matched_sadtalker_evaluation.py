"""Evaluate the matched real/SadTalker course pilot without test tuning.

The primary stability result uses repeated four-fold source-pair-grouped
cross-validation.  Every real/fake pair stays together, each repeat gives one
out-of-fold probability for all 24 clips, and all models use the same folds,
regularisation, class weighting and 0.5 decision threshold.

Four controlled representations are compared:

* context_free: global lip dynamics only;
* window_no_phone: the same local before/during/after windows, no phone labels;
* phone_context: local windows plus the observed neighbouring phone labels;
* shuffled_phone: identical features and model, but phone labels are permuted
  at source-pair level as a negative control.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from s2f.evaluate import records
from s2f.features import CONTEXT, build_features, read_intervals
from s2f.manifest import validate_manifest


ROOT = Path(__file__).resolve().parents[1]
SEED = 5230
REPEATS = 20
FOLDS = 4
THRESHOLD = 0.5


def grouped_folds(groups: pd.Series, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    unique = np.array(sorted(groups.unique()))
    if len(unique) < FOLDS or len(unique) % FOLDS:
        raise ValueError(f"Need a nonzero multiple of {FOLDS} source pairs")
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    result = []
    for test_groups in np.array_split(unique, FOLDS):
        test = groups.isin(test_groups).to_numpy()
        result.append((np.flatnonzero(~test), np.flatnonzero(test)))
    return result


def make_records(features: pd.DataFrame, mode: str) -> list[dict]:
    if mode in {"context_free", "phone_context", "shuffled_phone"}:
        return records(features, "context" if mode != "context_free" else "context_free")
    if mode != "window_no_phone":
        raise ValueError(f"Unknown mode: {mode}")
    rows = []
    for row in features.to_dict("records"):
        result = {
            key: float(value)
            for key, value in row.items()
            if key != "clip_id" and key not in CONTEXT
        }
        if not any(key.startswith("ctx_") for key in result):
            raise ValueError("window_no_phone requires context-window features")
        rows.append(result)
    return rows


def shuffled_phone_features(
    features: pd.DataFrame, manifest: pd.DataFrame, seed: int,
) -> pd.DataFrame:
    result = features.copy()
    pair_context = manifest[["source_id", "clip_id"]].merge(
        features[["clip_id"] + CONTEXT], on="clip_id", validate="one_to_one",
    )
    # The shared audio should give identical context inside each pair.  Failing
    # explicitly catches ASR/cache inconsistencies rather than hiding them.
    if (pair_context.groupby("source_id")[CONTEXT].nunique() > 1).any().any():
        raise ValueError("Real/fake pair received different weak phone labels")
    unique = pair_context.groupby("source_id", sort=True)[CONTEXT].first().reset_index()
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(unique))
    shuffled = unique.copy()
    shuffled[CONTEXT] = unique.loc[permutation, CONTEXT].to_numpy()
    mapping = shuffled.set_index("source_id")
    clip_to_source = manifest.set_index("clip_id").source_id
    for column in CONTEXT:
        result[column] = result.clip_id.map(clip_to_source).map(mapping[column])
    return result


def classifier() -> Pipeline:
    return Pipeline([
        ("vectorize", DictVectorizer(sparse=False)),
        ("scale", StandardScaler()),
        ("classifier", LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=2000, random_state=SEED,
        )),
    ])


def score(labels: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    prediction = probability >= THRESHOLD
    return {
        "macro_f1": float(f1_score(labels, prediction, average="macro", zero_division=0)),
        "auroc": float(roc_auc_score(labels, probability)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/matched_sadtalker_aligned_manifest.csv")
    parser.add_argument("--intervals", default="data/matched_sadtalker_intervals.csv")
    parser.add_argument("--features", default="outputs/matched_sadtalker/features")
    parser.add_argument("--out", default="artifacts/matched_sadtalker")
    args = parser.parse_args()

    manifest = validate_manifest(ROOT / args.manifest, ROOT, protocol="speaker")
    intervals = read_intervals(ROOT / args.intervals)
    features = build_features(manifest, ROOT / args.features, intervals=intervals)
    features = manifest[["clip_id"]].merge(features, on="clip_id", validate="one_to_one")
    labels = manifest.label.map({"real": 0, "fake": 1}).to_numpy(int)
    modes = ["context_free", "window_no_phone", "phone_context", "shuffled_phone"]
    fold_rows: list[dict] = []
    repeat_rows: list[dict] = []
    prediction_rows: list[dict] = []

    for repeat in range(REPEATS):
        mode_features = {
            "context_free": features,
            "window_no_phone": features,
            "phone_context": features,
            "shuffled_phone": shuffled_phone_features(features, manifest, SEED + repeat),
        }
        probabilities = {mode: np.full(len(manifest), np.nan) for mode in modes}
        for fold, (train_index, test_index) in enumerate(
            grouped_folds(manifest.source_id, SEED + repeat)
        ):
            train_groups = set(manifest.iloc[train_index].source_id)
            test_groups = set(manifest.iloc[test_index].source_id)
            if train_groups & test_groups:
                raise AssertionError("Source-pair leakage")
            for mode in modes:
                model_records = make_records(mode_features[mode], mode)
                model = classifier()
                model.fit([model_records[index] for index in train_index], labels[train_index])
                probability = model.predict_proba(
                    [model_records[index] for index in test_index]
                )[:, 1]
                probabilities[mode][test_index] = probability
                fold_rows.append({
                    "repeat": repeat,
                    "fold": fold,
                    "model": mode,
                    "n_train": int(len(train_index)),
                    "n_test": int(len(test_index)),
                    "train_pairs": int(len(train_groups)),
                    "test_pairs": int(len(test_groups)),
                    "n_features": int(len(
                        model.named_steps["vectorize"].get_feature_names_out()
                    )),
                    **score(labels[test_index], probability),
                })
        for mode in modes:
            if not np.isfinite(probabilities[mode]).all():
                raise AssertionError("Every clip needs one out-of-fold probability per repeat")
            repeat_rows.append({
                "repeat": repeat, "model": mode, "n": int(len(labels)),
                **score(labels, probabilities[mode]),
            })
            for index, probability in enumerate(probabilities[mode]):
                prediction_rows.append({
                    "repeat": repeat,
                    "model": mode,
                    "clip_id": manifest.iloc[index].clip_id,
                    "source_id": manifest.iloc[index].source_id,
                    "label": manifest.iloc[index].label,
                    "fake_probability": float(probability),
                    "prediction": int(probability >= THRESHOLD),
                })

    folds = pd.DataFrame(fold_rows)
    repeats = pd.DataFrame(repeat_rows)
    predictions = pd.DataFrame(prediction_rows)
    summary = repeats.groupby("model").agg(
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_sd=("macro_f1", "std"),
        auroc_mean=("auroc", "mean"),
        auroc_sd=("auroc", "std"),
    ).reset_index()
    pivot = repeats.pivot(index="repeat", columns="model", values=["macro_f1", "auroc"])
    comparisons = []
    comparison_pairs = [
        ("window_no_phone", "context_free"),
        ("phone_context", "context_free"),
        ("phone_context", "window_no_phone"),
        ("phone_context", "shuffled_phone"),
    ]
    for treatment, control in comparison_pairs:
        for metric in ["macro_f1", "auroc"]:
            delta = pivot[(metric, treatment)] - pivot[(metric, control)]
            comparisons.append({
                "comparison": f"{treatment}-minus-{control}",
                "metric": metric,
                "mean_delta": float(delta.mean()),
                "sd_delta": float(delta.std()),
                "positive_repeats": int((delta > 0).sum()),
                "ties": int((delta == 0).sum()),
                "negative_repeats": int((delta < 0).sum()),
            })
    comparisons = pd.DataFrame(comparisons)

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    folds.to_csv(out / "fold_metrics.csv", index=False)
    repeats.to_csv(out / "repeat_oof_metrics.csv", index=False)
    predictions.to_csv(out / "oof_predictions.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    comparisons.to_csv(out / "paired_comparisons.csv", index=False)
    manifest[["source_id"] + CONTEXT].drop_duplicates().to_csv(
        out / "pair_context_labels.csv", index=False,
    )
    feature_counts = folds.groupby("model").n_features.agg(["min", "max"]).to_dict("index")
    (out / "run.json").write_text(json.dumps({
        "seed": SEED, "repeats": REPEATS, "folds": FOLDS,
        "threshold": THRESHOLD, "classifier_C": 1.0,
        "group": "source_id (matched real/fake pair)",
        "models": modes,
        "feature_count_range_by_model": feature_counts,
        "interpretation": (
            "window_no_phone tests whether local temporal windows help without phone labels; "
            "phone_context tests the additional value of observed weak phone categories; "
            "shuffled_phone is a negative control for those categories."
        ),
        "claim_boundary": (
            "Small matched course pilot; repeated folds reuse the same 12 pairs and "
            "do not constitute independent datasets or publication evidence."
        ),
    }, indent=2), encoding="utf-8")

    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for axis, metric, title in [
        (axes[0], "macro_f1", "Out-of-fold Macro-F1"),
        (axes[1], "auroc", "Out-of-fold AUROC"),
    ]:
        data = [repeats.loc[repeats.model.eq(mode), metric] for mode in modes]
        axis.boxplot(data, tick_labels=["Global", "Window\nno phone", "Phone\ncontext", "Shuffled\nphone"])
        axis.set_ylim(0, 1.02)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Matched real/SadTalker pilot: 20 repeated 4-fold grouped CV")
    figure.tight_layout()
    figure.savefig(out / "matched_stability.png", dpi=180)
    plt.close(figure)

    print(summary.to_string(index=False))
    print(comparisons.to_string(index=False))
    print("outputs", out)


if __name__ == "__main__":
    main()
