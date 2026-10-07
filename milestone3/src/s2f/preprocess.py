"""Adapted from team M1 notebook cell 7; see reports/M1_PROVENANCE.md.

Keep M1 geometry/contours; correct pixel aspect ratio and never interpolate gaps.
One visible speaker is an input constraint, not biometric identity recognition.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .common import safe_id, sha256, write_json

OUTER = [61,146,91,181,84,17,314,405,321,375,291,409,270,269,0,37,39,40,185,61]
INNER = [78,95,88,178,87,14,317,402,318,324,308,415,310,311,13,82,81,80,191,78]
GEOMETRY = ["inner_aperture", "outer_aperture", "lip_width"]
DYNAMIC = ["aperture_velocity", "aperture_acceleration", "aperture_jerk"]


def get_video_metadata(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    info = {"fps": float(cap.get(cv2.CAP_PROP_FPS)),
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}
    cap.release()
    if not np.isfinite(info["fps"]) or info["fps"] <= 0 or info["frame_count"] < 1:
        raise ValueError("Invalid video FPS/frame count; transcode to constant frame rate")
    info["duration_s"] = info["frame_count"] / info["fps"]
    return info


def geometry(landmarks, width, height):
    def distance(a, b):
        pa, pb = landmarks[a], landmarks[b]
        return np.hypot((pa.x-pb.x)*width, (pa.y-pb.y)*height)
    scale = distance(33, 263)
    if scale < 1:
        return None
    return dict(zip(GEOMETRY, [distance(13,14)/scale, distance(0,17)/scale,
                               distance(61,291)/scale]))


def add_dynamics(df, window=5):
    if window < 1 or window % 2 == 0:
        raise ValueError("Smoothing window must be a positive odd integer")
    df = df.copy()
    t = df.time_s.to_numpy(float)
    if len(t) < 2 or not np.isfinite(t).all() or (np.diff(t) <= 0).any():
        raise ValueError("Need >=2 strictly increasing finite timestamps")
    for col in GEOMETRY:
        df[col + "_smooth"] = np.nan
    for col in DYNAMIC:
        df[col] = np.nan
    valid = df.detected.astype(bool).to_numpy() & np.isfinite(df[GEOMETRY]).all(axis=1).to_numpy()
    # Contiguous runs only: no smoothing or derivatives across missing detections.
    boundaries = np.r_[0, np.flatnonzero(valid[1:] != valid[:-1]) + 1, len(df)]
    for a,b in zip(boundaries[:-1], boundaries[1:]):
        if not valid[a]:
            continue
        ids = df.index[a:b]
        for col in GEOMETRY:
            df.loc[ids, col+"_smooth"] = df.loc[ids, col].rolling(window, center=True, min_periods=1).median()
        if b-a >= 7:
            x = df.loc[ids, "inner_aperture_smooth"].to_numpy(float)
            for col in DYNAMIC:
                x = np.gradient(x, t[a:b], edge_order=2)
                df.loc[ids, col] = x
            # Exclude three derivative boundary frames on both sides.
            df.loc[df.index[a:a+3], DYNAMIC] = np.nan
            df.loc[df.index[b-3:b], DYNAMIC] = np.nan
    return df


def plot_trajectory(df, path, title):
    fig, axes = plt.subplots(3, 1, figsize=(10,7), sharex=True)
    for ax,col,label in zip(axes, ["inner_aperture_smooth","lip_width_smooth","aperture_velocity"],
                            ["Aperture / eye distance","Width / eye distance","Aperture velocity (1/s)"]):
        ax.plot(df.time_s, df[col], linewidth=1.5)
        ax.set_ylabel(label)
        ax.grid(alpha=.2)
    axes[0].set_title(title)
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def preprocess(path, clip_id, model, out="outputs", stride=1, window=5):
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    safe_id(clip_id)
    if stride < 1:
        raise ValueError("stride must be >=1")
    info = get_video_metadata(path)
    out = Path(out)
    for folder in ["features", "figures", "metadata"]:
        (out/folder).mkdir(parents=True, exist_ok=True)
    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(model), delegate=python.BaseOptions.Delegate.CPU),
        running_mode=vision.RunningMode.VIDEO, num_faces=1,
        min_face_detection_confidence=.5, min_face_presence_confidence=.5, min_tracking_confidence=.5)
    rows, preview, idx, last_ms = [], False, 0, -1
    cap = cv2.VideoCapture(str(path))
    try:
        with vision.FaceLandmarker.create_from_options(options) as detector:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if idx % stride == 0:
                    # Presentation timestamps handle VFR; fallback to nominal FPS only if absent.
                    ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                    if not np.isfinite(ms) or ms <= last_ms:
                        ms = idx / info["fps"] * 1000
                    stamp = round(ms)
                    if stamp <= last_ms:
                        raise ValueError("Non-monotonic timestamps; convert video to CFR first")
                    last_ms = stamp
                    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    result = detector.detect_for_video(image, stamp)
                    row = {"frame_index":idx, "time_s":stamp/1000, "detected":False,
                           **{key:np.nan for key in GEOMETRY}}
                    if result.face_landmarks:
                        landmarks = result.face_landmarks[0]
                        values = geometry(landmarks, info["width"], info["height"])
                        if values:
                            row.update(values)
                            row["detected"] = True
                            if not preview:
                                for contour,color in [(OUTER,(0,255,0)),(INNER,(0,200,255))]:
                                    points = np.array([(int(landmarks[i].x*info["width"]),int(landmarks[i].y*info["height"])) for i in contour])
                                    cv2.polylines(frame, [points], False, color, 2)
                                cv2.imwrite(str(out/"figures"/f"{clip_id}_landmarks.png"), frame)
                                preview = True
                    rows.append(row)
                idx += 1
    finally:
        cap.release()
    if len(rows) < 2:
        raise ValueError("Insufficient decoded frames")
    if not preview:
        # A rerun of an existing clip ID must not display an older successful preview.
        (out/"figures"/f"{clip_id}_landmarks.png").unlink(missing_ok=True)
    df = add_dynamics(pd.DataFrame(rows), window)
    df.to_csv(out/"features"/f"{clip_id}.csv", index=False)
    info.update({"clip_id":clip_id, "decoded_frames":idx, "sampled_frames":len(df),
                 "coverage":float(df.detected.mean()), "frame_stride":stride,
                 "smoothing_window":window, "sha256":sha256(path), "model_sha256":sha256(model),
                 "status":"ok" if preview else "no_face", "timestamp_method":"presentation_ms_with_fps_fallback"})
    write_json(out/"metadata"/f"{clip_id}.json", info)
    plot_trajectory(df, out/"figures"/f"{clip_id}_trajectory.png", f"{clip_id}: lip dynamics (not a detection result)")
    return info
