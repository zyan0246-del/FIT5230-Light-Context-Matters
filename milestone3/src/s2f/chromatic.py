"""Paired forensic audit for periodic chromatic injection.

This module does not estimate physiology and does not decide whether a video is
real or fake.  It compares a control/modified pair and measures periodic colour
differences introduced by the modification.  A neutral re-encode pair should be
analysed with the same functions as a codec control.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd


DEFAULT_ROIS = {
    "global": (0.0, 0.0, 1.0, 1.0),
    "forehead": (0.32, 0.10, 0.68, 0.30),
    "left_cheek": (0.18, 0.38, 0.43, 0.68),
    "right_cheek": (0.57, 0.38, 0.82, 0.68),
}
CHANNELS = ("red", "green", "blue")


def _validate_roi(roi):
    if len(roi) != 4:
        raise ValueError("ROI must be (x0, y0, x1, y1)")
    x0, y0, x1, y1 = map(float, roi)
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        raise ValueError("ROI coordinates must satisfy 0 <= start < end <= 1")
    return x0, y0, x1, y1


def roi_channel_means(frame_bgr, roi):
    """Return RGB means from a normalized rectangular ROI in a BGR frame."""
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError("Expected an H x W x 3 frame")
    x0, y0, x1, y1 = _validate_roi(roi)
    height, width = frame_bgr.shape[:2]
    xa, xb = int(np.floor(x0 * width)), int(np.ceil(x1 * width))
    ya, yb = int(np.floor(y0 * height)), int(np.ceil(y1 * height))
    patch = frame_bgr[ya:yb, xa:xb]
    if patch.size == 0:
        raise ValueError("ROI produced an empty patch")
    b, g, r = patch.astype(np.float64).mean(axis=(0, 1))
    return {"red": float(r), "green": float(g), "blue": float(b)}


def spectral_features(signal, fps, low_hz=0.7, high_hz=3.0, target_hz=1.2,
                      target_half_width_hz=0.12):
    """Summarise periodic content after linear detrending and Hann windowing."""
    values = np.asarray(signal, dtype=float)
    if values.ndim != 1 or len(values) < 16 or not np.isfinite(values).all():
        raise ValueError("Signal must contain at least 16 finite samples")
    if not np.isfinite(fps) or fps <= 2 * high_hz:
        raise ValueError("FPS must exceed twice the upper frequency bound")
    if not (0 < low_hz < high_hz < fps / 2):
        raise ValueError("Invalid spectral band")
    time = np.arange(len(values), dtype=float)
    slope, intercept = np.polyfit(time, values, 1)
    detrended = values - (slope * time + intercept)
    windowed = detrended * np.hanning(len(values))
    frequency = np.fft.rfftfreq(len(values), d=1.0 / fps)
    power = np.abs(np.fft.rfft(windowed)) ** 2
    band = (frequency >= low_hz) & (frequency <= high_hz)
    if not band.any():
        raise ValueError("No FFT bin lies in the requested band")
    band_frequency = frequency[band]
    band_power = power[band]
    peak_index = int(np.argmax(band_power))
    peak_hz = float(band_frequency[peak_index])
    noise = np.delete(band_power, peak_index)
    noise_floor = float(np.median(noise)) if len(noise) else 0.0
    peak_power = float(band_power[peak_index])
    ratio = peak_power / max(noise_floor, np.finfo(float).eps)
    target = band & (np.abs(frequency - target_hz) <= target_half_width_hz)
    band_total = float(power[band].sum())
    target_fraction = float(power[target].sum() / band_total) if band_total > 0 else 0.0
    return {
        "n": int(len(values)),
        "fps": float(fps),
        "bin_width_hz": float(fps / len(values)),
        "peak_hz": peak_hz,
        "peak_bpm": peak_hz * 60.0,
        "peak_power": peak_power,
        "median_band_power": noise_floor,
        "peak_to_median_ratio": float(ratio),
        "target_hz": float(target_hz),
        "target_band_energy_fraction": target_fraction,
    }


def paired_colour_signals(control_path, modified_path, rois=None):
    """Decode aligned videos and return modified-minus-control ROI signals."""
    import cv2

    rois = dict(DEFAULT_ROIS if rois is None else rois)
    for roi in rois.values():
        _validate_roi(roi)
    controls = cv2.VideoCapture(str(control_path))
    modified = cv2.VideoCapture(str(modified_path))
    if not controls.isOpened() or not modified.isOpened():
        controls.release(); modified.release()
        raise ValueError("Could not open both videos")
    fps_a = float(controls.get(cv2.CAP_PROP_FPS))
    fps_b = float(modified.get(cv2.CAP_PROP_FPS))
    width_a = int(controls.get(cv2.CAP_PROP_FRAME_WIDTH))
    width_b = int(modified.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_a = int(controls.get(cv2.CAP_PROP_FRAME_HEIGHT))
    height_b = int(modified.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not np.isclose(fps_a, fps_b, atol=1e-6) or (width_a, height_a) != (width_b, height_b):
        controls.release(); modified.release()
        raise ValueError("Paired videos must share FPS and dimensions")
    rows = []
    index = 0
    try:
        while True:
            ok_a, frame_a = controls.read()
            ok_b, frame_b = modified.read()
            if ok_a != ok_b:
                raise ValueError("Paired videos decode to different frame counts")
            if not ok_a:
                break
            row = {"frame_index": index, "time_s": index / fps_a}
            for name, roi in rois.items():
                a = roi_channel_means(frame_a, roi)
                b = roi_channel_means(frame_b, roi)
                for channel in CHANNELS:
                    row[f"{name}_{channel}_delta"] = b[channel] - a[channel]
            rows.append(row)
            index += 1
    finally:
        controls.release(); modified.release()
    if len(rows) < 16:
        raise ValueError("Insufficient aligned decoded frames")
    return pd.DataFrame(rows), fps_a


def analyse_pair(control_path, modified_path, out, expected_hz=1.2, rois=None):
    """Run the paired audit and save signals plus a JSON spectral summary."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    frame, fps = paired_colour_signals(control_path, modified_path, rois)
    frame.to_csv(out / "paired_colour_signals.csv", index=False)
    summaries = []
    for column in [c for c in frame if c.endswith("_delta")]:
        result = spectral_features(frame[column], fps, target_hz=expected_hz)
        roi, channel, _ = column.rsplit("_", 2)
        summaries.append({"roi": roi, "channel": channel, **result})
    summary = {
        "control": str(control_path),
        "modified": str(modified_path),
        "interpretation": "paired chromatic difference; not an rPPG or real/fake score",
        "spectra": summaries,
    }
    (out / "spectral_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )
    pd.DataFrame(summaries).to_csv(out / "spectral_summary.csv", index=False)
    return summary
