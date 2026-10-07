"""Pair external rPPG probabilities with our independently computed lip scores."""
import numpy as np
import pandas as pd


def compare_pairs(pair_path,lip_predictions,out,alpha=None):
    pairs=pd.read_csv(pair_path,keep_default_na=False,dtype=str)
    needed={"pair_id","variant","clip_id","rppg_fake_probability"}
    if not needed<=set(pairs):
        raise ValueError(f"Pair table requires {sorted(needed)}")
    if pairs.clip_id.duplicated().any() or pairs.pair_id.str.strip().eq("").any():
        raise ValueError("Unique clip IDs and nonempty pair IDs required")
    for _,group in pairs.groupby("pair_id"):
        if len(group)!=2 or set(group.variant)!={"original","modified"}:
            raise ValueError("Each pair must have one original and one modified")
    lip=pd.read_csv(lip_predictions)
    if lip.clip_id.duplicated().any() or not set(pairs.clip_id)<=set(lip.clip_id):
        raise ValueError("Missing/duplicate lip predictions")
    frame=pairs.merge(lip[["clip_id","fake_probability"]],on="clip_id",validate="one_to_one").rename(columns={"fake_probability":"lip_fake_probability"})
    raw=frame.rppg_fake_probability
    frame["rppg_fake_probability"]=pd.to_numeric(raw.replace("",np.nan),errors="raise")
    for column in ["rppg_fake_probability","lip_fake_probability"]:
        values=pd.to_numeric(frame[column],errors="raise")
        present=values.dropna()
        if not np.isfinite(present).all() or not present.between(0,1).all() or (column.startswith("lip") and values.isna().any()):
            raise ValueError(f"{column} requires probabilities in [0,1]")
    if alpha is not None:
        if not np.isfinite(alpha) or not 0<=alpha<=1:
            raise ValueError("Fusion alpha must be in [0,1]")
        # Missing rPPG => missing fused score, never substitute zero.
        frame["fused_fake_probability"]=alpha*frame.lip_fake_probability+(1-alpha)*frame.rppg_fake_probability
    for column in [c for c in frame if c.endswith("fake_probability")]:
        baseline=frame[frame.variant.eq("original")].set_index("pair_id")[column]
        frame[column+"_delta_from_original"]=frame[column]-frame.pair_id.map(baseline)
    frame.to_csv(out,index=False)
    return frame
