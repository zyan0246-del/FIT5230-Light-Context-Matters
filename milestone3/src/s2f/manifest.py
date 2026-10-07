"""Explicit splits, content hashes and source families prevent clip leakage."""
from pathlib import Path
import pandas as pd
from .common import safe_id, sha256

REQUIRED = ["clip_id", "file_path", "label", "speaker_id", "generator", "transcript",
            "central_phoneme", "previous_phoneme", "next_phoneme", "compression",
            "split", "source_id"]


def validate_manifest(path, root=".", protocol="speaker", check_files=True):
    if protocol not in {"speaker", "generator", "speaker_generator", "clip"}:
        raise ValueError("Unknown evaluation protocol")
    df = pd.read_csv(path, keep_default_na=False, dtype=str)
    missing = set(REQUIRED) - set(df.columns)
    if missing:
        raise ValueError(f"Missing manifest columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("Manifest is empty")
    for col in REQUIRED:
        if df[col].str.strip().eq("").any():
            raise ValueError(f"Blank {col}; use 'unknown' for unannotated phonemes/transcript")
    for value in df.clip_id:
        safe_id(value)
    if df.clip_id.duplicated().any():
        raise ValueError("Duplicate clip_id")
    if not set(df.label) <= {"real", "fake"}:
        raise ValueError("label must be real/fake")
    if not set(df.split) <= {"train", "val", "test"}:
        raise ValueError("split must be train/val/test")
    paths = [(Path(root) / p).resolve() for p in df.file_path]
    if len(set(paths)) != len(paths):
        raise ValueError("Duplicate file paths")
    if check_files:
        absent = [str(p) for p in paths if not p.is_file()]
        if absent:
            raise ValueError(f"Missing files: {absent}")
        hashes = [sha256(p) for p in paths]
        if len(set(hashes)) != len(hashes):
            raise ValueError("Duplicate clip contents (SHA256), even if renamed")
        df["sha256"] = hashes
    groups = ["source_id"]
    if protocol in {"speaker", "speaker_generator"}:
        if df.speaker_id.isin(["unknown", "n/a"]).any():
            raise ValueError("Speaker-disjoint evaluation requires known speaker IDs")
        groups.append("speaker_id")
    for col in groups:
        if (df.groupby(col).split.nunique() > 1).any():
            raise ValueError(f"Train/val/test leakage through {col}")
    if protocol in {"generator", "speaker_generator"}:
        fakes = df[df.label.eq("fake")]
        if fakes.generator.isin(["unknown", "n/a"]).any():
            raise ValueError("Generator protocol requires known fake generators")
        if (fakes.groupby("generator").split.nunique() > 1).any():
            raise ValueError("Train/val/test leakage through fake generator")
    return df
