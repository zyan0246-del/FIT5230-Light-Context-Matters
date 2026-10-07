"""Run the paired chromatic audit under fixed post-processing conditions."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-m3")
import matplotlib.pyplot as plt
import pandas as pd

from s2f.chromatic import analyse_pair


ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / "data/private/dark_mimic"
WORK = ROOT / "outputs/m3_work/chromatic_robustness"
CONTROL = PRIVATE / "person01_fake01_control.mp4"
ATTACK = PRIVATE / "person01_fake01_attacked_A1.mp4"
NEUTRAL = PRIVATE / "person01_fake01_control_neutral_reencode.mp4"
CONDITIONS = {
    "original": [],
    "h264_crf35": ["-vf", "scale=512:448", "-r", "25", "-c:v", "libx264", "-crf", "35", "-preset", "veryfast"],
    "resize_256": ["-vf", "scale=256:224", "-r", "25", "-c:v", "libx264", "-crf", "23", "-preset", "veryfast"],
    "fps_15": ["-vf", "fps=15,scale=512:448", "-r", "15", "-c:v", "libx264", "-crf", "23", "-preset", "veryfast"],
}


def transcode(source: Path, destination: Path, options: list[str]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
        *options, "-pix_fmt", "yuv420p", "-an", str(destination),
    ]
    subprocess.run(command, check=True)


def path_for(source: Path, condition: str, role: str) -> Path:
    if condition == "original":
        return source
    destination = WORK / "videos" / condition / f"{role}.mp4"
    transcode(source, destination, CONDITIONS[condition])
    return destination


def main() -> None:
    rows = []
    for condition in CONDITIONS:
        control = path_for(CONTROL, condition, "control")
        attack = path_for(ATTACK, condition, "attack")
        neutral = path_for(NEUTRAL, condition, "neutral")
        for comparison, modified in [("attack", attack), ("neutral", neutral)]:
            out = WORK / comparison / condition
            analyse_pair(control, modified, out, expected_hz=1.2)
            summary = pd.read_csv(out / "spectral_summary.csv")
            global_rows = summary[summary.roi.eq("global")]
            for record in global_rows.to_dict("records"):
                rows.append({
                    "condition": condition,
                    "comparison": comparison,
                    "channel": record["channel"],
                    "fps": record["fps"],
                    "n_frames": record["n"],
                    "peak_hz": record["peak_hz"],
                    "peak_to_median_ratio": record["peak_to_median_ratio"],
                    "target_band_energy_fraction": record["target_band_energy_fraction"],
                })

    frame = pd.DataFrame(rows)
    table = frame.pivot(index=["condition", "channel"], columns="comparison",
                        values="target_band_energy_fraction").reset_index()
    table["attack_to_neutral_ratio"] = table.attack / table.neutral
    table.to_csv(WORK / "robustness_summary.csv", index=False)

    red = table[table.channel.eq("red")].set_index("condition").loc[list(CONDITIONS)]
    x = range(len(red))
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar([i - 0.18 for i in x], red.attack, width=0.36, label="Dark.Mimic A=1")
    ax.bar([i + 0.18 for i in x], red.neutral, width=0.36, label="Neutral re-encode")
    ax.set_xticks(list(x), ["Original", "H.264 CRF 35", "256×224", "15 FPS"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Red-channel energy fraction near 1.2 Hz")
    ax.set_title("Periodic chromatic trace under post-processing")
    ax.grid(axis="y", alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(WORK / "robustness_red.png", dpi=180)
    plt.close(fig)

    report = {
        "conditions_fixed_before_analysis": list(CONDITIONS),
        "expected_frequency_hz": 1.2,
        "scope": "one paired Dark.Mimic A=1 example; descriptive forensic audit",
        "red_channel": red.reset_index().to_dict("records"),
        "limitations": [
            "The paired audit requires an aligned control.",
            "This is not an rPPG estimator or a population-level detector evaluation.",
            "Only one peer-supplied identity/pair was available.",
        ],
    }
    (WORK / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
