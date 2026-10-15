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
import re
import sys
from pathlib import Path

import cmudict
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
from s2f.features import build_features  # noqa: E402


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


def residual_events(events: pd.DataFrame, real_training_ids: set[str]) -> pd.DataFrame:
    numeric = [f"{column}_{phase}" for column in GEOMETRY for phase in ["before", "after"]]
    training = events.clip_id.isin(real_training_ids)
    if not training.any():
        raise ValueError("Expected-context model needs real training events")
    result = events.copy()
    for column in GEOMETRY:
        preprocessing = ColumnTransformer([
            ("category", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ("numeric", StandardScaler(), numeric),
        ])
        model = Pipeline([
            ("preprocess", preprocessing),
            ("ridge", Ridge(alpha=RIDGE_ALPHA)),
        ])
        model.fit(
            events.loc[training, CATEGORICAL + numeric],
            events.loc[training, f"{column}_during"],
        )
        prediction = model.predict(events[CATEGORICAL + numeric])
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

    # Repeated outer-fold stability comparison. Context expectation is refit
    # inside each fold using only real clips in that outer training fold.
    global_features = build_features(manifest, ROOT / "outputs/m3_work/pilot_features/features")
    global_table = manifest[["clip_id", "label"]].merge(global_features, on="clip_id", validate="one_to_one")
    global_columns = [column for column in global_features if column != "clip_id"]
    cross_validator = RepeatedStratifiedKFold(
        n_splits=CV_SPLITS, n_repeats=CV_REPEATS, random_state=SEED,
    )
    fold_rows: list[dict] = []
    prediction_rows: list[pd.DataFrame] = []
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
    fig.suptitle("Multi-event context: promising split, no stable average gain")
    fig.tight_layout()
    fig.savefig(artifact / "multi_context_improvement.png", dpi=180)
    plt.close(fig)

    counts = events.groupby("clip_id").size()
    fallback_clips = sorted(events.loc[
        events.selection_threshold.eq(FALLBACK_WORD_PROBABILITY), "clip_id"
    ].unique().tolist())
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
        "interpretation": (
            "The original six-clip split improves, but repeated cross-validation does not show a stable "
            "average gain. Treat the result as a feasibility improvement and motivate a fresh matched hold-out set."
        ),
    }
    (artifact / "multi_context_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
