"""Context v1: time-window dynamics multiplied by phonetic-context indicators.

Not CALS/PIA reproduction and not automatic alignment. One central interval per
clip in v1 deliberately avoids treating correlated intervals as independent data.
"""
from typing import Protocol
from pathlib import Path
import numpy as np
import pandas as pd
from .preprocess import GEOMETRY, DYNAMIC
from .common import safe_id

SIGNALS = [c+"_smooth" for c in GEOMETRY] + DYNAMIC
CONTEXT = ["previous_phoneme", "central_phoneme", "next_phoneme"]


class Aligner(Protocol):
    def align(self, video_path: Path, transcript: str) -> pd.DataFrame:
        """Return interval-schema rows; an automatic implementation is future work."""
        ...


def stats(df, prefix=""):
    result = {}
    for col in SIGNALS:
        x = df[col].to_numpy(float)
        x = x[np.isfinite(x)]
        if len(x) < 2:
            raise ValueError(f"Insufficient valid values in {prefix}{col}")
        for name,value in [("mean",np.mean(x)),("std",np.std(x)),("abs_mean",np.mean(abs(x))),
                           ("p95_abs",np.percentile(abs(x),95))]:
            result[prefix+col+"_"+name] = float(value)
    return result


def read_intervals(path):
    df = pd.read_csv(path, keep_default_na=False)
    required = ["clip_id", "start_s", "end_s"] + CONTEXT + ["annotation_source"]
    if set(required)-set(df.columns):
        raise ValueError(f"Intervals require {required}")
    if df.clip_id.duplicated().any():
        raise ValueError("v1 supports exactly one central interval per clip")
    for col in ["start_s","end_s"]:
        df[col] = pd.to_numeric(df[col], errors="raise")
        if not np.isfinite(df[col]).all():
            raise ValueError("Interval timestamps must be finite")
    if (df.start_s < 0).any() or (df.end_s <= df.start_s).any():
        raise ValueError("Intervals require 0 <= start_s < end_s")
    for col in CONTEXT + ["annotation_source"]:
        if df[col].astype(str).str.strip().isin(["", "unknown"]).any():
            raise ValueError(f"Context intervals require annotated {col}")
    return df.set_index("clip_id")


def build_features(manifest, directory, intervals=None, coverage=.8, context_window=.2, min_frames=5):
    rows = []
    for clip in manifest.itertuples():
        safe_id(clip.clip_id)
        df = pd.read_csv(Path(directory)/f"{clip.clip_id}.csv")
        valid = df.detected.astype(str).str.lower().isin(["true","1"])
        if valid.mean() < coverage:
            raise ValueError(f"{clip.clip_id}: coverage {valid.mean():.3f} < {coverage}; excluded, never labelled real")
        result = {"clip_id":clip.clip_id, **stats(df)}
        if intervals is not None:
            if clip.clip_id not in intervals.index:
                raise ValueError(f"Missing interval for {clip.clip_id}; same clips required for paired comparison")
            interval = intervals.loc[clip.clip_id]
            a,b = float(interval.start_s),float(interval.end_s)
            if a-context_window < df.time_s.min() or b+context_window > df.time_s.max():
                raise ValueError(f"{clip.clip_id}: interval/context outside sampled duration")
            for col in CONTEXT:
                if str(getattr(clip,col)) != str(interval[col]):
                    raise ValueError(f"{clip.clip_id}: manifest/interval mismatch for {col}")
                result[col] = str(interval[col])
            for name,lo,hi in [("before",a-context_window,a),("during",a,b),("after",b,b+context_window)]:
                part = df[(df.time_s>=lo)&(df.time_s<hi)]
                part_valid = valid.loc[part.index]
                if len(part)<min_frames or part_valid.mean()<coverage:
                    raise ValueError(f"{clip.clip_id}: insufficient {name} context coverage/frames")
                result.update(stats(part, "ctx_"+name+"_"))
            for col in SIGNALS:
                result[f"ctx_delta_{col}"] = result[f"ctx_after_{col}_mean"]-result[f"ctx_before_{col}_mean"]
        rows.append(result)
    return pd.DataFrame(rows)
