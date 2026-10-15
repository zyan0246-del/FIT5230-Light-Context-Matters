"""Exploratory multi-event context follow-up for the M3 lip experiment.

This analysis was designed after the original six-clip test result was known.
It is therefore explicitly post-hoc and cannot turn that test set into fresh
confirmatory evidence.  The fixed-split result is accompanied by a repeated
four-fold stability check so that a favourable single split is not hidden.

The upgrade addresses two concrete limitations of Context v1:

* use every auditable central-vowel proxy instead of one interval per clip;
* learn expected central mouth geometry from neighbouring phone classes and
  real-training speech, then aggregate residuals at clip level.

No held-out clip contributes to the expected-context model in either the fixed
split or an outer cross-validation fold.  Correlated events are never treated
as independent classification examples: classification remains clip-level.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


SEED = 5230
MIN_WORD_PROBABILITY = 0.80
FALLBACK_WORD_PROBABILITY = 0.60
CENTRAL_HALF_WINDOW_S = 0.10
CONTEXT_WINDOW_S = 0.20
RIDGE_ALPHA = 10.0
CLASSIFIER_C = 0.01
CV_SPLITS = 4
CV_REPEATS = 20

WORD_RE = re.compile(r"[A-Za-z']+")
VOWELS = {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"}
PHONE_GROUPS = {
    "vowel": VOWELS,
    "stop": {"P", "B", "T", "D", "K", "G"},
    "fricative": {"F", "V", "TH", "DH", "S", "Z", "SH", "ZH", "HH"},
    "affricate": {"CH", "JH"},
    "nasal": {"M", "N", "NG"},
    "liquid": {"L", "R"},
    "glide": {"W", "Y"},
}
GEOMETRY = ["inner_aperture_smooth", "outer_aperture_smooth", "lip_width_smooth"]
CATEGORICAL = ["previous_class", "next_class", "central_vowel"]


def clean_phone(phone: str) -> str:
    return re.sub(r"\d", "", str(phone).upper())


def phone_class(phone: str) -> str:
    phone = clean_phone(phone)
    for name, group in PHONE_GROUPS.items():
        if phone in group:
            return name
    return "other"


def flatten_words(result: dict, dictionary: dict) -> list[dict]:
    phones: list[dict] = []
    for segment in result.get("segments", []):
        for word in segment.get("words", []):
            match = WORD_RE.search(word.get("word", ""))
            token = match.group(0).lower() if match else ""
            if token not in dictionary:
                continue
            start, end = float(word["start"]), float(word["end"])
            pronunciation = dictionary[token][0]
            if end <= start or not pronunciation:
                continue
            step = (end - start) / len(pronunciation)
            for index, raw_phone in enumerate(pronunciation):
                phones.append({
                    "word": token,
                    "word_probability": float(word.get("probability", 0.0)),
                    "phone": clean_phone(raw_phone),
                    "start_s": start + index * step,
                    "end_s": start + (index + 1) * step,
                })
    return phones


def _phase_mean(frame: pd.DataFrame, start: float, end: float, column: str) -> float:
    selected = frame.loc[
        (frame.time_s >= start) & (frame.time_s < end) & frame.detected,
        column,
    ].to_numpy(float)
    selected = selected[np.isfinite(selected)]
    return float(np.mean(selected)) if len(selected) >= 4 else np.nan


def clip_events(clip_id: str, dictionary: dict) -> list[dict]:
    asr_path = ROOT / "outputs/m3_work/asr" / f"{clip_id}.json"
    feature_path = ROOT / "outputs/m3_work/pilot_features/features" / f"{clip_id}.csv"
    if not asr_path.exists() or not feature_path.exists():
        raise FileNotFoundError("Run the M3 ASR and landmark cells before this follow-up")
    phones = flatten_words(json.loads(asr_path.read_text(encoding="utf-8")), dictionary)
    frame = pd.read_csv(feature_path)
    frame["detected"] = frame.detected.astype(str).str.lower().isin(["true", "1"])

    # Robust within-clip normalisation reduces identity and face-scale effects.
    for column in GEOMETRY:
        values = frame.loc[frame.detected, column].to_numpy(float)
        median = float(np.nanmedian(values))
        scale = float(np.nanpercentile(values, 75) - np.nanpercentile(values, 25))
        if not np.isfinite(scale) or scale < 1e-6:
            scale = float(np.nanstd(values))
        if not np.isfinite(scale) or scale < 1e-6:
            scale = 1.0
        frame[column + "_z"] = (frame[column] - median) / scale

    minimum_time, maximum_time = float(frame.time_s.min()), float(frame.time_s.max())
    # An ASR rerun can place every word in a short clip below 0.80. Relax the
    # same rule for any clip with zero usable events and record that decision.
    # The threshold is never selected using class labels or evaluation scores.
    for threshold in (MIN_WORD_PROBABILITY, FALLBACK_WORD_PROBABILITY):
        events: list[dict] = []
        for index in range(1, len(phones) - 1):
            central = phones[index]
            if central["phone"] not in VOWELS or central["word_probability"] < threshold:
                continue
            centre = (central["start_s"] + central["end_s"]) / 2
            start, end = centre - CENTRAL_HALF_WINDOW_S, centre + CENTRAL_HALF_WINDOW_S
            if start - CONTEXT_WINDOW_S < minimum_time or end + CONTEXT_WINDOW_S > maximum_time:
                continue
            event = {
                "clip_id": clip_id,
                "event_index": len(events),
                "start_s": start,
                "end_s": end,
                "central_word": central["word"],
                "word_probability": central["word_probability"],
                "selection_threshold": threshold,
                "previous_class": phone_class(phones[index - 1]["phone"]),
                "next_class": phone_class(phones[index + 1]["phone"]),
                "central_vowel": central["phone"],
            }
            valid = True
            for column in GEOMETRY:
                normalised = column + "_z"
                for phase, lower, upper in [
                    ("before", start - CONTEXT_WINDOW_S, start),
                    ("during", start, end),
                    ("after", end, end + CONTEXT_WINDOW_S),
                ]:
                    value = _phase_mean(frame, lower, upper, normalised)
                    event[f"{column}_{phase}"] = value
                    valid &= np.isfinite(value)
            if valid:
                events.append(event)
        if events:
            return events
    raise ValueError(f"{clip_id}: no eligible multi-context event at either confidence threshold")


def residual_events(
    events: pd.DataFrame, real_training_ids: set[str], *, use_phone_context: bool = True,
) -> pd.DataFrame:
    numeric = [f"{column}_{phase}" for column in GEOMETRY for phase in ["before", "after"]]
    training = events.clip_id.isin(real_training_ids)
    if not training.any():
        raise ValueError("Expected-context model needs real training events")
    result = events.copy()
    for column in GEOMETRY:
        transformers = [("numeric", StandardScaler(), numeric)]
        inputs = numeric
        if use_phone_context:
            transformers.insert(0, ("category", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL))
            inputs = CATEGORICAL + numeric
        preprocessing = ColumnTransformer(transformers)
        model = Pipeline([
            ("preprocess", preprocessing),
            ("ridge", Ridge(alpha=RIDGE_ALPHA)),
        ])
        model.fit(
            events.loc[training, inputs],
            events.loc[training, f"{column}_during"],
        )
        prediction = model.predict(events[inputs])
        result[f"residual_{column}"] = events[f"{column}_during"] - prediction
    return result


def aggregate_events(events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for clip_id, group in events.groupby("clip_id", sort=True):
        row: dict[str, float | str] = {"clip_id": clip_id, "event_count": int(len(group))}
        for column in GEOMETRY:
            absolute_residual = np.abs(group[f"residual_{column}"].to_numpy(float))
            row[f"residual_{column}_mean"] = float(np.mean(absolute_residual))
            row[f"residual_{column}_p90"] = float(np.percentile(absolute_residual, 90))
            transition_sets = {
                "in": group[f"{column}_during"] - group[f"{column}_before"],
                "out": group[f"{column}_after"] - group[f"{column}_during"],
                "curvature": (
                    group[f"{column}_before"] + group[f"{column}_after"]
                    - 2 * group[f"{column}_during"]
                ),
            }
            for name, values in transition_sets.items():
                row[f"{column}_{name}_abs_mean"] = float(np.abs(values).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def classifier() -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("classifier", LogisticRegression(
            C=CLASSIFIER_C, class_weight="balanced", max_iter=2000, random_state=SEED,
        )),
    ])


def score(labels: np.ndarray, probability: np.ndarray) -> dict:
    return {
        "n": int(len(labels)),
        "macro_f1": float(f1_score(labels, probability >= 0.5, average="macro")),
        "auroc": float(roc_auc_score(labels, probability)),
    }


def main() -> None:
    output = ROOT / "outputs/m3_work/multi_context_followup"
    artifact = ROOT / "artifacts/m3_improvement"
    output.mkdir(parents=True, exist_ok=True)
    artifact.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(ROOT / "data/m3_pilot_aligned_manifest.csv", keep_default_na=False)
    reuse_events = os.environ.get("M3_REUSE_AUDITED_EVENTS") == "1"
    if reuse_events:
        cache = output / "multi_context_events.csv"
        if not cache.is_file():
            raise FileNotFoundError(f"Audited event cache missing: {cache}")
        events = pd.read_csv(cache)
        if set(events.clip_id) != set(manifest.clip_id):
            raise ValueError("Cached events do not cover the manifest exactly")
        expected = manifest.set_index("clip_id")[["label", "split", "speaker_id", "source_id"]]
        for field in expected.columns:
            if not all(events[field] == events.clip_id.map(expected[field])):
                raise ValueError(f"Cached {field} differs from current manifest")
    else:
        import cmudict
        dictionary = cmudict.dict()
        events = pd.DataFrame([
            event for clip_id in manifest.clip_id for event in clip_events(clip_id, dictionary)
        ])
        events = events.merge(
            manifest[["clip_id", "label", "split", "speaker_id", "source_id"]],
            on="clip_id", validate="many_to_one",
        )
    events.to_csv(output / "multi_context_events.csv", index=False)

    labels = manifest.label.map({"real": 0, "fake": 1}).to_numpy()
    training_mask = manifest.split.eq("train").to_numpy()
    real_training_ids = set(manifest.loc[manifest.split.eq("train") & manifest.label.eq("real"), "clip_id"])
    fixed_events = residual_events(events, real_training_ids)
    fixed_features = aggregate_events(fixed_events)
    table = manifest[["clip_id", "label", "split"]].merge(fixed_features, on="clip_id", validate="one_to_one")
    feature_columns = [column for column in fixed_features if column not in {"clip_id", "event_count"}]
    model = classifier()
    model.fit(table.loc[training_mask, feature_columns], labels[training_mask])

    fixed_rows: list[dict] = []
    fixed_predictions: list[pd.DataFrame] = []
    for split in ["val", "test"]:
        mask = table.split.eq(split).to_numpy()
        probability = model.predict_proba(table.loc[mask, feature_columns])[:, 1]
        fixed_rows.append({"model": "multi_event_context", "split": split, **score(labels[mask], probability)})
        frame = table.loc[mask, ["clip_id", "label", "event_count"]].copy()
        frame["fake_probability"] = probability
        frame["prediction"] = (probability >= 0.5).astype(int)
        fixed_predictions.append(frame)
    fixed_metrics = pd.DataFrame(fixed_rows)
    fixed_metrics.to_csv(artifact / "fixed_split_metrics.csv", index=False)
    pd.concat(fixed_predictions, ignore_index=True).to_csv(artifact / "fixed_split_predictions.csv", index=False)
    fixed_features.to_csv(artifact / "multi_event_clip_features.csv", index=False)

    # Post-hoc controlled ablation: keep windows, numerical inputs, ridge alpha,
    # aggregation, classifier and split fixed; remove only phonetic categories.
    no_phone_events = residual_events(events, real_training_ids, use_phone_context=False)
    no_phone_features = aggregate_events(no_phone_events)
    no_phone_table = manifest[["clip_id", "label", "split"]].merge(
        no_phone_features, on="clip_id", validate="one_to_one",
    )
    no_phone_columns = [column for column in no_phone_features if column not in {"clip_id", "event_count"}]
    no_phone_model = classifier()
    no_phone_model.fit(no_phone_table.loc[training_mask, no_phone_columns], labels[training_mask])
    ablation_fixed_rows = []
    for split in ["val", "test"]:
        mask = no_phone_table.split.eq(split).to_numpy()
        probability = no_phone_model.predict_proba(no_phone_table.loc[mask, no_phone_columns])[:, 1]
        ablation_fixed_rows.append({"model": "multi_event_no_phone", "split": split, **score(labels[mask], probability)})
    pd.DataFrame(fixed_rows + ablation_fixed_rows).to_csv(
        artifact / "matched_context_ablation_fixed.csv", index=False,
    )

    # Repeated outer-fold stability comparison. Context expectation is refit
    # inside each fold using only real clips in that outer training fold.
    if reuse_events:
        cached_features = pd.read_csv(ROOT / "outputs/m3_work/pilot_eval_context/metrics/model_features.csv")
        global_features = cached_features[[
            column for column in cached_features
            if column == "clip_id" or (not column.startswith("ctx_") and not column.endswith("_phoneme"))
        ]]
        if set(global_features.clip_id) != set(manifest.clip_id):
            raise ValueError("Cached feature rows do not cover the manifest exactly")
    else:
        from s2f.features import build_features
        global_features = build_features(manifest, ROOT / "outputs/m3_work/pilot_features/features")
    global_table = manifest[["clip_id", "label"]].merge(global_features, on="clip_id", validate="one_to_one")
    global_columns = [column for column in global_features if column != "clip_id"]
    cross_validator = RepeatedStratifiedKFold(
        n_splits=CV_SPLITS, n_repeats=CV_REPEATS, random_state=SEED,
    )
    fold_rows: list[dict] = []
    prediction_rows: list[pd.DataFrame] = []
    ablation_fold_rows: list[dict] = []
    for fold, (train_index, test_index) in enumerate(cross_validator.split(global_table, labels)):
        baseline = Pipeline([
            ("scale", StandardScaler()),
            ("classifier", LogisticRegression(
                C=1.0, class_weight="balanced", max_iter=2000, random_state=SEED,
            )),
        ])
        baseline.fit(global_table.iloc[train_index][global_columns], labels[train_index])
        baseline_probability = baseline.predict_proba(global_table.iloc[test_index][global_columns])[:, 1]
        fold_rows.append({"fold": fold, "model": "context_free", **score(labels[test_index], baseline_probability)})

        outer_train_ids = set(global_table.iloc[train_index].clip_id)
        outer_real_ids = set(
            global_table.iloc[train_index].loc[
                global_table.iloc[train_index].label.eq("real"), "clip_id"
            ]
        )
        fold_events = residual_events(events, outer_real_ids)
        fold_features = aggregate_events(fold_events)
        fold_table = global_table[["clip_id"]].merge(fold_features, on="clip_id", validate="one_to_one")
        fold_columns = [column for column in fold_features if column not in {"clip_id", "event_count"}]
        improved = classifier()
        improved.fit(fold_table.iloc[train_index][fold_columns], labels[train_index])
        improved_probability = improved.predict_proba(fold_table.iloc[test_index][fold_columns])[:, 1]
        fold_rows.append({"fold": fold, "model": "multi_event_context", **score(labels[test_index], improved_probability)})

        fold_no_phone = aggregate_events(
            residual_events(events, outer_real_ids, use_phone_context=False)
        )
        fold_no_phone_table = global_table[["clip_id"]].merge(
            fold_no_phone, on="clip_id", validate="one_to_one",
        )
        fold_no_phone_columns = [
            column for column in fold_no_phone if column not in {"clip_id", "event_count"}
        ]
        fold_no_phone_model = classifier()
        fold_no_phone_model.fit(
            fold_no_phone_table.iloc[train_index][fold_no_phone_columns], labels[train_index],
        )
        fold_no_phone_probability = fold_no_phone_model.predict_proba(
            fold_no_phone_table.iloc[test_index][fold_no_phone_columns],
        )[:, 1]
        ablation_fold_rows.extend([
            {"fold": fold, "model": "multi_event_context", **score(labels[test_index], improved_probability)},
            {"fold": fold, "model": "multi_event_no_phone", **score(labels[test_index], fold_no_phone_probability)},
        ])

        for name, probability in [
            ("context_free", baseline_probability),
            ("multi_event_context", improved_probability),
        ]:
            prediction_rows.append(pd.DataFrame({
                "fold": fold,
                "repeat": fold // CV_SPLITS,
                "model": name,
                "clip_id": global_table.iloc[test_index].clip_id.to_numpy(),
                "label": labels[test_index],
                "fake_probability": probability,
            }))

    fold_metrics = pd.DataFrame(fold_rows)
    fold_metrics.to_csv(artifact / "repeated_cv_fold_metrics.csv", index=False)
    ablation_fold_metrics = pd.DataFrame(ablation_fold_rows)
    ablation_fold_metrics.to_csv(artifact / "matched_context_ablation_folds.csv", index=False)
    ablation_summary = ablation_fold_metrics.groupby("model")[["macro_f1", "auroc"]].mean()
    ablation_summary.to_csv(artifact / "matched_context_ablation_summary.csv")
    pd.concat(prediction_rows, ignore_index=True).to_csv(artifact / "repeated_cv_predictions.csv", index=False)
    cv_summary = (
        fold_metrics.groupby("model")[["macro_f1", "auroc"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    cv_summary.columns = ["model", "macro_f1_mean", "macro_f1_std", "auroc_mean", "auroc_std"]
    cv_summary.to_csv(artifact / "repeated_cv_summary.csv", index=False)

    pivot = fold_metrics.pivot(index="fold", columns="model", values=["macro_f1", "auroc"])
    paired = {
        metric: {
            "mean_delta": float((pivot[metric].multi_event_context - pivot[metric].context_free).mean()),
            "median_delta": float((pivot[metric].multi_event_context - pivot[metric].context_free).median()),
            "positive_fold_fraction": float((pivot[metric].multi_event_context > pivot[metric].context_free).mean()),
        }
        for metric in ["macro_f1", "auroc"]
    }

    original = pd.read_csv(ROOT / "outputs/m3_work/pilot_eval_context/metrics/results.csv")
    original_test = original[original.split.eq("test")][["model", "macro_f1", "auroc"]].copy()
    original_test["model"] = original_test.model.replace({"context_free": "Context-free", "context": "Single weak context"})
    improved_test = fixed_metrics[fixed_metrics.split.eq("test")][["model", "macro_f1", "auroc"]].copy()
    improved_test["model"] = "Multi-event context"
    plot_data = pd.concat([original_test, improved_test], ignore_index=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    x = np.arange(len(plot_data))
    axes[0].bar(x - 0.18, plot_data.macro_f1, 0.36, label="Macro-F1")
    axes[0].bar(x + 0.18, plot_data.auroc, 0.36, label="AUROC")
    axes[0].set_xticks(x, plot_data.model, rotation=12, ha="right")
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Original held-out split (n=6, post-hoc follow-up)")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.2)

    cv_plot = cv_summary.set_index("model").loc[["context_free", "multi_event_context"]]
    labels_plot = ["Context-free", "Multi-event context"]
    x = np.arange(2)
    axes[1].bar(x - 0.18, cv_plot.macro_f1_mean, 0.36, yerr=cv_plot.macro_f1_std, label="Macro-F1")
    axes[1].bar(x + 0.18, cv_plot.auroc_mean, 0.36, yerr=cv_plot.auroc_std, label="AUROC")
    axes[1].set_xticks(x, labels_plot)
    axes[1].set_ylim(0, 1)
    axes[1].set_title("20× repeated four-fold stability check")
    axes[1].grid(axis="y", alpha=0.2)
    axes[1].legend()
    fig.suptitle("Multi-event context: post-hoc split and stability check")
    fig.tight_layout()
    fig.savefig(artifact / "multi_context_improvement.png", dpi=180)
    plt.close(fig)

    counts = events.groupby("clip_id").size()
    fallback_clips = sorted(events.loc[
        events.selection_threshold.eq(FALLBACK_WORD_PROBABILITY), "clip_id"
    ].unique().tolist())
    baseline_test = original_test.loc[original_test.model.eq("Context-free")].iloc[0]
    followup_test = improved_test.iloc[0]
    fixed_f1_delta = float(followup_test.macro_f1 - baseline_test.macro_f1)
    fixed_auc_delta = float(followup_test.auroc - baseline_test.auroc)
    cv_f1_delta = float(paired["macro_f1"]["mean_delta"])
    cv_auc_delta = float(paired["auroc"]["mean_delta"])
    fixed_description = (
        "improved both metrics" if fixed_f1_delta > 0 and fixed_auc_delta > 0
        else "did not improve both metrics"
    )
    cv_description = (
        "did not show a consistent gain on both metrics"
        if cv_f1_delta <= 0 or cv_auc_delta <= 0
        else "showed a positive mean change on both metrics, but needs independent confirmation"
    )
    summary = {
        "status": "exploratory_post_hoc",
        "parameters": {
            "seed": SEED,
            "minimum_word_probability": MIN_WORD_PROBABILITY,
            "fallback_word_probability_if_no_events": FALLBACK_WORD_PROBABILITY,
            "central_window_s": 2 * CENTRAL_HALF_WINDOW_S,
            "context_window_each_side_s": CONTEXT_WINDOW_S,
            "ridge_alpha": RIDGE_ALPHA,
            "classifier_c": CLASSIFIER_C,
            "cv_splits": CV_SPLITS,
            "cv_repeats": CV_REPEATS,
        },
        "events": {
            "total": int(len(events)),
            "clips": int(counts.size),
            "minimum_per_clip": int(counts.min()),
            "median_per_clip": float(counts.median()),
            "maximum_per_clip": int(counts.max()),
            "fallback_clips": fallback_clips,
        },
        "fixed_split_metrics": fixed_rows,
        "repeated_cv": cv_summary.to_dict("records"),
        "paired_fold_differences": paired,
        "matched_context_ablation": {
            "fixed_split": fixed_rows + ablation_fixed_rows,
            "repeated_cv_mean": ablation_summary.reset_index().to_dict("records"),
            "caveat": "Post-hoc same-setting comparison. Real training-event residuals remain in sample; independent data is needed.",
        },
        "interpretation": (
            f"In this runtime, the post-hoc six-clip follow-up {fixed_description} "
            f"(Macro-F1 delta {fixed_f1_delta:+.3f}, AUROC delta {fixed_auc_delta:+.3f}). "
            f"Repeated cross-validation {cv_description} "
            f"(mean Macro-F1 delta {cv_f1_delta:+.3f}, AUROC delta {cv_auc_delta:+.3f}). "
            "This is exploratory and requires a fresh matched hold-out set."
        ),
    }
    (artifact / "multi_context_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
