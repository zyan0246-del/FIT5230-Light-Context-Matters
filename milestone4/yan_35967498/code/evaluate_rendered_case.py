"""Re-extracted detector and decoded-RGB checks for one rendered attack case."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


def feature_stats(frame: pd.DataFrame) -> dict[str, float]:
    columns = [
        "inner_aperture_smooth", "outer_aperture_smooth", "lip_width_smooth",
        "aperture_velocity", "aperture_acceleration", "aperture_jerk",
    ]
    record: dict[str, float] = {}
    for column in columns:
        values = frame[column].to_numpy(float)
        values = values[np.isfinite(values)]
        for name, value in (
            ("mean", np.mean(values)),
            ("std", np.std(values)),
            ("abs_mean", np.mean(np.abs(values))),
            ("p95_abs", np.percentile(np.abs(values), 95)),
        ):
            record[f"{column}_{name}"] = float(value)
    return record


def decoded_rgb_psnr(original: Path, attacked: Path) -> tuple[int, float]:
    first, second = cv2.VideoCapture(str(original)), cv2.VideoCapture(str(attacked))
    squared_error, samples, frames = 0.0, 0, 0
    while True:
        ok_a, frame_a = first.read()
        ok_b, frame_b = second.read()
        if not (ok_a and ok_b):
            break
        difference = frame_a.astype(np.float64) - frame_b.astype(np.float64)
        squared_error += float(np.square(difference).sum())
        samples += difference.size
        frames += 1
    first.release()
    second.release()
    if not frames or not samples:
        raise ValueError("Videos could not be compared")
    mse = squared_error / samples
    return frames, float(10 * np.log10(255**2 / mse))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-features", type=Path, required=True)
    parser.add_argument("--clip-id", required=True)
    parser.add_argument("--original-features", type=Path, required=True)
    parser.add_argument("--attacked-features", type=Path, required=True)
    parser.add_argument("--original-video", type=Path, required=True)
    parser.add_argument("--attacked-video", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    base = pd.read_csv(args.base_features)
    feature_columns = [column for column in base if column not in {"clip_id", "label"}]
    labels = base.label.map({"real": 0, "fake": 1}).to_numpy(int)
    matches = base.index[base.clip_id.eq(args.clip_id)]
    if len(matches) != 1 or base.loc[matches[0], "label"] != "fake":
        raise ValueError("clip-id must identify exactly one fake base row")
    index = int(matches[0])
    original = feature_stats(pd.read_csv(args.original_features))
    attacked = feature_stats(pd.read_csv(args.attacked_features))

    rows = []
    for repeat in range(20):
        splitter = StratifiedKFold(4, shuffle=True, random_state=5230 + repeat)
        for fold, (train, test) in enumerate(splitter.split(base[feature_columns], labels)):
            if index not in test:
                continue
            scaler = StandardScaler().fit(base.loc[train, feature_columns])
            model = LogisticRegression(
                C=0.1, class_weight="balanced", max_iter=2000, random_state=5230
            ).fit(scaler.transform(base.loc[train, feature_columns]), labels[train])
            original_probability = float(
                model.predict_proba(
                    scaler.transform(pd.DataFrame([original])[feature_columns])
                )[0, 1]
            )
            attacked_probability = float(
                model.predict_proba(
                    scaler.transform(pd.DataFrame([attacked])[feature_columns])
                )[0, 1]
            )
            rows.append(
                {
                    "repeat": repeat,
                    "fold": fold,
                    "original_fake_probability": original_probability,
                    "attacked_fake_probability": attacked_probability,
                    "probability_drop": original_probability - attacked_probability,
                    "original_evades": original_probability < 0.5,
                    "attacked_evades": attacked_probability < 0.5,
                }
            )
    predictions = pd.DataFrame(rows)
    frames, psnr = decoded_rgb_psnr(args.original_video, args.attacked_video)
    render_metadata_path = args.attacked_video.with_suffix(".json")
    render_metadata = (
        json.loads(render_metadata_path.read_text())
        if render_metadata_path.exists()
        else {}
    )
    summary = {
        "clip_id": args.clip_id,
        "prespecified_case": True,
        "decoded_frames_compared": frames,
        "decoded_rgb_psnr_db": psnr,
        "landmark_coverage_during_render": render_metadata.get("landmark_coverage"),
        "mean_absolute_scale_change": render_metadata.get(
            "mean_absolute_scale_change"
        ),
        "maximum_absolute_scale_change": render_metadata.get(
            "maximum_absolute_scale_change"
        ),
        "attacked_video_sha256": render_metadata.get("output_sha256"),
        "n_repeats": len(predictions),
        "mean_original_fake_probability": float(
            predictions.original_fake_probability.mean()
        ),
        "mean_attacked_fake_probability": float(
            predictions.attacked_fake_probability.mean()
        ),
        "mean_probability_drop": float(predictions.probability_drop.mean()),
        "positive_drop_repeats": int((predictions.probability_drop > 0).sum()),
        "original_evasion_fraction": float(predictions.original_evades.mean()),
        "attacked_evasion_fraction": float(predictions.attacked_evades.mean()),
        "claim_boundary": (
            "One prespecified rendered clip. It establishes render-and-reextract "
            "feasibility, not reliable evasion, imperceptibility, or generalisation."
        ),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.out / "detector_predictions.csv", index=False)
    (args.out / "detector_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
