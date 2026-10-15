"""Test whether non-physiological periodic lighting triggers the colour metric.

This is an artificial global-brightness perturbation on public real videos,
not a naturally captured lighting challenge and not an attack on the peer pair.
Both the periodic and neutral variants use identical codec settings.
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
FREQUENCY_HZ = 1.2
BRIGHTNESS_AMPLITUDE = 0.02


def encode(source: Path, target: Path, periodic: bool) -> list[str]:
    target.parent.mkdir(parents=True, exist_ok=True)
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
               "-map", "0:v:0"]
    if periodic:
        command += ["-vf", "eq=brightness='0.02*sin(2*PI*1.2*t)':eval=frame"]
    command += ["-c:v", "mpeg4", "-b:v", "800k", "-pix_fmt", "yuv420p", "-an", str(target)]
    subprocess.run(command, check=True)
    return command


def metric(pair_out: Path) -> tuple[float, float, float]:
    summary = pd.read_csv(pair_out / "spectral_summary.csv")
    row = summary[summary.roi.eq("global") & summary.channel.eq("red")].iloc[0]
    signal = pd.read_csv(pair_out / "paired_colour_signals.csv")
    rms = float(np.sqrt(np.mean(np.square(signal.global_red_delta))))
    return float(row.target_band_energy_fraction), rms, float(row.peak_hz)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-dir", type=Path, default=ROOT / "data/processed/m3_pilot")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/m3_work/periodic_light")
    args = parser.parse_args()
    real = pd.read_csv(ROOT / "data/m3_pilot_manifest.csv")
    real = real[real.label.eq("real")]
    if real.speaker_id.nunique() != len(real):
        raise ValueError("Expected distinct public real-speaker IDs")
    rows = []
    for item in real.itertuples(index=False):
        source = args.video_dir / f"{item.clip_id}.mp4"
        if not source.is_file():
            raise FileNotFoundError(source)
        neutral = args.out / "videos" / f"{item.clip_id}_neutral.mp4"
        periodic = args.out / "videos" / f"{item.clip_id}_periodic.mp4"
        encode(source, neutral, periodic=False)
        encode(source, periodic, periodic=True)
        neutral_out = args.out / "pairs" / item.clip_id / "neutral"
        periodic_out = args.out / "pairs" / item.clip_id / "periodic"
        analyse_pair(source, neutral, neutral_out, expected_hz=FREQUENCY_HZ)
        analyse_pair(source, periodic, periodic_out, expected_hz=FREQUENCY_HZ)
        neutral_fraction, neutral_rms, neutral_peak = metric(neutral_out)
        periodic_fraction, periodic_rms, periodic_peak = metric(periodic_out)
        rows.append({
            "clip_id": item.clip_id,
            "speaker_id": item.speaker_id,
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "neutral_fraction": neutral_fraction,
            "periodic_fraction": periodic_fraction,
            "neutral_red_delta_rms": neutral_rms,
            "periodic_red_delta_rms": periodic_rms,
            "neutral_peak_hz": neutral_peak,
            "periodic_peak_hz": periodic_peak,
        })
    args.out.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows).sort_values("clip_id")
    table.to_csv(args.out / "periodic_light_controls.csv", index=False)
    report = {
        "n_speakers": int(table.speaker_id.nunique()),
        "frequency_hz": FREQUENCY_HZ,
        "global_brightness_amplitude": BRIGHTNESS_AMPLITUDE,
        "neutral_command_template": "ffmpeg -i SOURCE -map 0:v:0 -c:v mpeg4 -b:v 800k -pix_fmt yuv420p -an NEUTRAL",
        "periodic_command_template": "ffmpeg -i SOURCE -map 0:v:0 -vf eq=brightness='0.02*sin(2*PI*1.2*t)':eval=frame -c:v mpeg4 -b:v 800k -pix_fmt yuv420p -an PERIODIC",
        "neutral_fraction_median": float(table.neutral_fraction.median()),
        "periodic_fraction_median": float(table.periodic_fraction.median()),
        "periodic_fraction_min": float(table.periodic_fraction.min()),
        "periodic_fraction_max": float(table.periodic_fraction.max()),
        "periodic_higher_than_neutral_count": int((table.periodic_fraction > table.neutral_fraction).sum()),
        "limitations": [
            "The brightness modulation is synthetic and global, not naturally captured lighting.",
            "The test has no attacked multi-identity sample and does not estimate detector specificity.",
            "The measured metric is a paired chromatic difference, not an rPPG waveform.",
        ],
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(table[["clip_id", "neutral_fraction", "periodic_fraction", "periodic_red_delta_rms"]].to_string(index=False))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
