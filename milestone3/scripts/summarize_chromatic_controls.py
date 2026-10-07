"""Compare the attack pair with a neutral re-encode control after both audits."""
from pathlib import Path
import json
import os

os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs/.mplcache").resolve()))
import matplotlib.pyplot as plt
import pandas as pd


ATTACK = Path("outputs/m3_work/dark_mimic_attack")
NEUTRAL = Path("outputs/m3_work/dark_mimic_neutral")
OUT = Path("outputs/m3_work/chromatic_comparison")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    attack = pd.read_csv(ATTACK / "spectral_summary.csv")
    neutral = pd.read_csv(NEUTRAL / "spectral_summary.csv")
    keys = ["roi", "channel"]
    cols = ["peak_hz", "peak_to_median_ratio", "target_band_energy_fraction"]
    joined = attack[keys + cols].merge(
        neutral[keys + cols], on=keys, suffixes=("_attack", "_neutral"), validate="one_to_one"
    )
    for col in ["peak_to_median_ratio", "target_band_energy_fraction"]:
        joined[col + "_ratio"] = joined[col + "_attack"] / joined[col + "_neutral"]
    joined.to_csv(OUT / "attack_vs_neutral.csv", index=False)

    signals = pd.read_csv(ATTACK / "paired_colour_signals.csv")
    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    for channel, colour in [("red", "#C43C39"), ("green", "#2B8A3E"), ("blue", "#2D5FB3")]:
        axes[0].plot(
            signals.time_s,
            signals[f"global_{channel}_delta"],
            label=channel,
            color=colour,
            linewidth=1.3,
        )
    axes[0].set_title("Dark.Mimic attack minus control: global decoded-pixel differences")
    axes[0].set_ylabel("Mean delta (0-255 units)")
    axes[0].set_xlabel("Time (s)")
    axes[0].grid(alpha=0.2)
    axes[0].legend(ncol=3)

    global_rows = joined[joined.roi.eq("global")].set_index("channel")
    x = range(len(global_rows))
    axes[1].bar(
        [v - 0.18 for v in x],
        global_rows.target_band_energy_fraction_attack,
        width=0.36,
        label="A=1 attack",
        color="#C46A2D",
    )
    axes[1].bar(
        [v + 0.18 for v in x],
        global_rows.target_band_energy_fraction_neutral,
        width=0.36,
        label="neutral re-encode",
        color="#5C7C99",
    )
    axes[1].set_xticks(list(x), global_rows.index)
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Energy fraction near 1.2 Hz")
    axes[1].set_title("Matched codec control separates periodic injection from re-encoding")
    axes[1].grid(axis="y", alpha=0.2)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(OUT / "attack_vs_neutral.png", dpi=180)
    plt.close(fig)

    report = {
        "scope": "one Dark.Mimic pair plus one neutral re-encode control",
        "classification": "not performed",
        "main_observation": {
            channel: {
                "attack_peak_hz": float(row.peak_hz_attack),
                "attack_target_energy_fraction": float(row.target_band_energy_fraction_attack),
                "neutral_target_energy_fraction": float(row.target_band_energy_fraction_neutral),
                "energy_fraction_ratio": float(row.target_band_energy_fraction_ratio),
            }
            for channel, row in global_rows.iterrows()
        },
        "limitations": [
            "single source pair",
            "rectangular image-relative ROIs are not face tracked",
            "paired audit requires a control video",
            "neutral control matches codec family and target bitrate, not the peer's exact encoder settings",
            "periodicity is not a physiological waveform or a real/fake decision",
        ],
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(joined.to_string(index=False))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
