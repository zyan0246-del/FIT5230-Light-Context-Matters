"""Summarise the held-out lip pilot with paired stratified bootstrap uncertainty."""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-m3")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "outputs/m3_work/pilot_eval_context"
OUT = ROOT / "outputs/m3_work/lip_pilot_summary"


def scores(labels: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    prediction = (probabilities >= 0.5).astype(int)
    f1 = f1_score(labels, prediction, labels=[0, 1], average="macro", zero_division=0)
    auc = roc_auc_score(labels, probabilities)
    return float(f1), float(auc)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    free = pd.read_csv(EVAL / "predictions/context_free_test.csv").sort_values("clip_id").reset_index(drop=True)
    context = pd.read_csv(EVAL / "predictions/context_test.csv").sort_values("clip_id").reset_index(drop=True)
    if free.clip_id.tolist() != context.clip_id.tolist() or free.label.tolist() != context.label.tolist():
        raise ValueError("Paired model predictions do not describe identical held-out clips")
    labels = free.label.map({"real": 0, "fake": 1}).to_numpy()
    probabilities = {
        "context_free": free.fake_probability.to_numpy(float),
        "weak_context": context.fake_probability.to_numpy(float),
    }
    point = {name: scores(labels, values) for name, values in probabilities.items()}

    rng = np.random.default_rng(5230)
    boot = {name: [] for name in probabilities}
    deltas = []
    by_class = [np.flatnonzero(labels == value) for value in [0, 1]]
    for _ in range(5000):
        sample = np.concatenate([rng.choice(indices, size=len(indices), replace=True) for indices in by_class])
        iteration = {name: scores(labels[sample], values[sample]) for name, values in probabilities.items()}
        for name, value in iteration.items():
            boot[name].append(value)
        deltas.append((iteration["weak_context"][0] - iteration["context_free"][0],
                       iteration["weak_context"][1] - iteration["context_free"][1]))

    rows = []
    for name in probabilities:
        values = np.asarray(boot[name])
        rows.append({
            "model": name,
            "n_test": len(labels),
            "macro_f1": point[name][0],
            "macro_f1_ci_low": np.percentile(values[:, 0], 2.5),
            "macro_f1_ci_high": np.percentile(values[:, 0], 97.5),
            "auroc": point[name][1],
            "auroc_ci_low": np.percentile(values[:, 1], 2.5),
            "auroc_ci_high": np.percentile(values[:, 1], 97.5),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "test_metrics_with_bootstrap.csv", index=False)
    delta = np.asarray(deltas)
    delta_report = {
        "weak_context_minus_context_free": {
            "macro_f1": float(point["weak_context"][0] - point["context_free"][0]),
            "macro_f1_bootstrap_95_ci": [float(np.percentile(delta[:, 0], 2.5)), float(np.percentile(delta[:, 0], 97.5))],
            "auroc": float(point["weak_context"][1] - point["context_free"][1]),
            "auroc_bootstrap_95_ci": [float(np.percentile(delta[:, 1], 2.5)), float(np.percentile(delta[:, 1], 97.5))],
        },
        "bootstrap": "5000 paired stratified resamples; fixed threshold 0.5",
        "interpretation": "No evidence that weak proxy context improves the six-clip held-out pilot.",
        "limitations": [
            "Only six held-out clips (three real, three fake).",
            "Real and fake clips come from different source collections, so domain cues may confound the pilot.",
            "Whisper word timestamps plus uniform CMUdict phone placement are weak proxies, not forced alignment.",
            "The result is feasibility evidence, not a publication-level generalisation claim.",
        ],
    }
    (OUT / "report.json").write_text(json.dumps(delta_report, indent=2), encoding="utf-8")

    failures = free[["clip_id", "label", "generator"]].copy()
    failures["context_free_probability"] = free.fake_probability
    failures["context_probability"] = context.fake_probability
    failures["context_free_error"] = ((free.fake_probability >= 0.5).astype(int) != labels)
    failures["context_error"] = ((context.fake_probability >= 0.5).astype(int) != labels)
    failures.to_csv(OUT / "held_out_predictions_and_failures.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    positions = np.arange(2)
    labels_display = ["Context-free", "Weak phonetic context"]
    for ax, metric, title in [(axes[0], "macro_f1", "Macro-F1"), (axes[1], "auroc", "AUROC")]:
        values = summary[metric].to_numpy()
        low = values - summary[f"{metric}_ci_low"].to_numpy()
        high = summary[f"{metric}_ci_high"].to_numpy() - values
        ax.bar(positions, values, color=["#4C78A8", "#E45756"], width=0.62)
        ax.errorbar(positions, values, yerr=np.vstack([low, high]), fmt="none", color="black", capsize=4)
        ax.set_xticks(positions, labels_display, rotation=10)
        ax.set_ylim(0, 1.05)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Held-out public pilot (n=6): weak context does not improve the baseline")
    fig.tight_layout()
    fig.savefig(OUT / "lip_pilot_results.png", dpi=180)
    plt.close(fig)
    print(summary.to_string(index=False))
    print(json.dumps(delta_report, indent=2))


if __name__ == "__main__":
    main()
