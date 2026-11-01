"""Render one landmark-guided lip-dynamics perturbation into an MP4.

The transform amplifies frame-to-frame aperture deviations around a rolling
median while keeping each local vertical scale bounded. It is a renderability
case study, not an imperceptibility guarantee or a general detector evasion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


MOUTH = sorted(set([61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
                    409, 270, 269, 0, 37, 39, 40, 185, 78, 95, 88, 178,
                    87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 13,
                    82, 81, 80, 191]))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def detect(frames: list[np.ndarray], fps: float, model: Path) -> list[np.ndarray | None]:
    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=str(model), delegate=python.BaseOptions.Delegate.CPU
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    results: list[np.ndarray | None] = []
    with vision.FaceLandmarker.create_from_options(options) as detector:
        for index, frame in enumerate(frames):
            image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            )
            result = detector.detect_for_video(image, round(index / fps * 1000))
            if not result.face_landmarks:
                results.append(None)
                continue
            landmarks = result.face_landmarks[0]
            height, width = frame.shape[:2]
            results.append(
                np.asarray([(point.x * width, point.y * height) for point in landmarks])
            )
    return results


def warp_mouth(frame: np.ndarray, points: np.ndarray, scale: float) -> np.ndarray:
    mouth = points[MOUTH]
    x0, y0 = mouth.min(axis=0)
    x1, y1 = mouth.max(axis=0)
    width, height = x1 - x0, y1 - y0
    x0 = max(0, int(np.floor(x0 - 0.35 * width)))
    x1 = min(frame.shape[1], int(np.ceil(x1 + 0.35 * width)))
    y0 = max(0, int(np.floor(y0 - 0.75 * height)))
    y1 = min(frame.shape[0], int(np.ceil(y1 + 0.75 * height)))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return frame

    roi = frame[y0:y1, x0:x1]
    centre_y = float((points[13, 1] + points[14, 1]) / 2 - y0)
    grid_x, grid_y = np.meshgrid(
        np.arange(roi.shape[1], dtype=np.float32),
        np.arange(roi.shape[0], dtype=np.float32),
    )
    map_y = centre_y + (grid_y - centre_y) / float(scale)
    warped = cv2.remap(
        roi,
        grid_x,
        map_y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    yy = (grid_y - roi.shape[0] / 2) / max(roi.shape[0] / 2, 1)
    xx = (grid_x - roi.shape[1] / 2) / max(roi.shape[1] / 2, 1)
    mask = np.clip(1 - xx**2 - yy**2, 0, 1) ** 2
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=2.0)[..., None]
    output = frame.copy()
    output[y0:y1, x0:x1] = np.clip(
        mask * warped + (1 - mask) * roi, 0, 255
    ).astype(np.uint8)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--gain", type=float, default=1.35)
    parser.add_argument("--max-scale-change", type=float, default=0.12)
    args = parser.parse_args()

    capture = cv2.VideoCapture(str(args.input))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames: list[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if not frames or not np.isfinite(fps) or fps <= 0:
        raise ValueError("Input video could not be decoded")

    landmarks = detect(frames, fps, args.model)
    aperture = np.full(len(frames), np.nan)
    for index, points in enumerate(landmarks):
        if points is not None:
            aperture[index] = np.linalg.norm(points[13] - points[14])
    valid = np.isfinite(aperture) & (aperture > 0.5)
    if valid.mean() < 0.9:
        raise ValueError(f"Landmark coverage {valid.mean():.3f} is below 0.9")
    filled = pd.Series(aperture).interpolate(limit_direction="both")
    baseline = filled.rolling(9, center=True, min_periods=1).median().to_numpy()
    target = baseline + args.gain * (filled.to_numpy() - baseline)
    limit = float(args.max_scale_change)
    scales = np.clip(target / filled.to_numpy(), 1 - limit, 1 + limit)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="fit5230-render-") as temporary:
        silent = Path(temporary) / "silent.mp4"
        height, width = frames[0].shape[:2]
        writer = cv2.VideoWriter(
            str(silent), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        for frame, points, scale in zip(frames, landmarks, scales):
            writer.write(frame if points is None else warp_mouth(frame, points, scale))
        writer.release()
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error", "-i", str(silent),
                "-i", str(args.input), "-map", "0:v:0", "-map", "1:a?",
                "-c:v", "libx264", "-crf", "23", "-preset", "medium",
                "-c:a", "aac", "-shortest", str(args.output),
            ],
            check=True,
        )

    record = {
        "input_sha256": sha256(args.input),
        "output_sha256": sha256(args.output),
        "frames": len(frames),
        "fps": fps,
        "landmark_coverage": float(valid.mean()),
        "gain": args.gain,
        "max_scale_change": limit,
        "mean_absolute_scale_change": float(np.mean(np.abs(scales - 1))),
        "maximum_absolute_scale_change": float(np.max(np.abs(scales - 1))),
        "claim_boundary": (
            "One prespecified rendered case. Visual quality, detector evasion, "
            "and generalisation require separate measurements."
        ),
    }
    args.output.with_suffix(".json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
