"""Measure the paired chromatic metric on public, unmodified real speakers.

Each source is compared with its own MPEG-4 re-encode. This controls ordinary
codec differences across speakers; it is not an attack or a lighting challenge.
The transcoded videos and frame-level traces remain in ignored outputs/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from s2f.chromatic import analyse_pair


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def transcode_neutral(source: Path, target: Path) -> list[str]:
    target.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
        "-map", "0:v:0", "-c:v", "mpeg4", "-b:v", "800k", "-pix_fmt", "yuv420p",
        "-an", str(target),
    ]
    subprocess.run(command, check=True)
    return command


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-dir", type=Path, default=ROOT / "data/processed/m3_pilot")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/m3_work/benign_chromatic")
    args = parser.parse_args()
    manifest = pd.read_csv(ROOT / "data/m3_pilot_manifest.csv")
    real = manifest[manifest.label.eq("real")]
    if real.speaker_id.nunique() != len(real):
        raise ValueError("Expected distinct real-video speaker IDs")

    rows = []
    commands = []
    for item in real.itertuples(index=False):
        source = args.video_dir / f"{item.clip_id}.mp4"
        if not source.is_file():
            raise FileNotFoundError(source)
        neutral = args.out / "videos" / f"{item.clip_id}_neutral.mp4"
        commands.append(transcode_neutral(source, neutral))
        pair_out = args.out / "pairs" / item.clip_id
        analyse_pair(source, neutral, pair_out, expected_hz=1.2)
        summary = pd.read_csv(pair_out / "spectral_summary.csv")
        signal = pd.read_csv(pair_out / "paired_colour_signals.csv")
        row = summary[summary.roi.eq("global") & summary.channel.eq("red")].iloc[0]
        rows.append({
            "clip_id": item.clip_id,
            "speaker_id": item.speaker_id,
            "source_sha256": sha256(source),
            "neutral_sha256": sha256(neutral),
            "n_frames": int(row.n),
            "fps": float(row.fps),
            "peak_hz": float(row.peak_hz),
            "target_band_energy_fraction": float(row.target_band_energy_fraction),
            "red_delta_rms": float(np.sqrt(np.mean(np.square(signal.global_red_delta)))),
        })
    args.out.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows).sort_values("clip_id")
    frame.to_csv(args.out / "benign_controls.csv", index=False)
    report = {
        "scope": "12 independent public real speakers, each paired with its own neutral MPEG-4 re-encode",
        "target_frequency_hz": 1.2,
        "command_template": "ffmpeg -i SOURCE -map 0:v:0 -c:v mpeg4 -b:v 800k -pix_fmt yuv420p -an TARGET",
        "n_speakers": int(frame.speaker_id.nunique()),
        "n_pairs": int(len(frame)),
        "red_target_fraction_min": float(frame.target_band_energy_fraction.min()),
        "red_target_fraction_median": float(frame.target_band_energy_fraction.median()),
        "red_target_fraction_max": float(frame.target_band_energy_fraction.max()),
        "red_delta_rms_median": float(frame.red_delta_rms.median()),
        "limitations": [
            "The videos are not matched to the Dark.Mimic identity or encoder settings.",
            "No controlled periodic lighting, head motion or exposure perturbations are included.",
            "No attack is applied to these speakers; this is not a false-positive-rate or detector test.",
            "A normalized spectral fraction can be large even if absolute signal amplitude is small.",
        ],
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(frame[["clip_id", "target_band_energy_fraction", "red_delta_rms"]].to_string(index=False))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
