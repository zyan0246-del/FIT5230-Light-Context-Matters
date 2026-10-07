"""Create basic metadata QC and a middle-frame contact sheet for the M3 pilot."""
from pathlib import Path
import json
import os
import subprocess

os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs/.mplcache").resolve()))
import cv2
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "m3_pilot_manifest.csv"
OUT = ROOT / "outputs" / "m3_work" / "pilot_qc"


def has_audio(path):
    command = [
        "ffprobe", "-v", "error", "-select_streams", "a:0",
        "-show_entries", "stream=codec_name", "-of", "json", str(path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    return bool(json.loads(result.stdout).get("streams"))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(MANIFEST, keep_default_na=False)
    records, images = [], []
    for row in manifest.itertuples():
        path = ROOT / row.file_path
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise ValueError(f"Cannot open {path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, frames // 2))
        ok, frame = capture.read()
        capture.release()
        if not ok:
            raise ValueError(f"Cannot read middle frame from {path}")
        images.append((row.clip_id, row.label, row.split, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        records.append({
            "clip_id": row.clip_id,
            "label": row.label,
            "split": row.split,
            "fps": fps,
            "frames": frames,
            "duration_s": frames / fps,
            "width": width,
            "height": height,
            "has_audio": has_audio(path),
            "bytes": path.stat().st_size,
        })
    qc = pd.DataFrame(records)
    qc.to_csv(OUT / "metadata_qc.csv", index=False)
    fig, axes = plt.subplots(6, 4, figsize=(12, 16))
    for ax, (clip_id, label, split, image) in zip(axes.ravel(), images):
        ax.imshow(image)
        ax.set_title(f"{clip_id}\n{label} / {split}", fontsize=8)
        ax.axis("off")
    fig.suptitle("M3 24-clip pilot: middle-frame visual audit", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(OUT / "contact_sheet.png", dpi=170)
    plt.close(fig)
    print(qc.groupby(["split", "label"])[["duration_s", "has_audio"]].agg(["count", "mean"]))
    print("contact sheet", (OUT / "contact_sheet.png").relative_to(ROOT))


if __name__ == "__main__":
    main()
