"""Strict 70/20/10 scoring, preserving the M1 challenge contract."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from .common import write_json

FIELDS=["clip_id","prediction","fake_probability","start_s","end_s","cue","comment"]
CUES={"context_transition","timing","motion_smoothness","visual_artifact","other"}


def load_catalog(path):
    df=pd.read_csv(path,keep_default_na=False,dtype={"clip_id":str})
    if not {"clip_id","duration_s"}<=set(df.columns) or df.empty or df.clip_id.duplicated().any():
        raise ValueError("Catalog needs unique clip_id and duration_s")
    df["duration_s"]=pd.to_numeric(df.duration_s,errors="raise")
    if not np.isfinite(df.duration_s).all() or (df.duration_s<=0).any():
        raise ValueError("Invalid catalog duration")
    return df.set_index("clip_id")


def validate_submission(path,catalog_path):
    df=pd.read_csv(path,keep_default_na=False,dtype=str)
    if set(FIELDS)-set(df.columns):
        raise ValueError(f"Required columns: {FIELDS}")
    catalog=load_catalog(catalog_path)
    if df.clip_id.duplicated().any():
        raise ValueError("Duplicate submission clip_id")
    if set(df.clip_id)!=set(catalog.index):
        raise ValueError(f"Missing clips: {sorted(set(catalog.index)-set(df.clip_id))}; unknown clips: {sorted(set(df.clip_id)-set(catalog.index))}")
    errors=[]
    for row in df.itertuples():
        try:
            if row.prediction not in {"0","1"}:
                raise ValueError("prediction must be integer 0/1")
            p=float(row.fake_probability)
            if not np.isfinite(p) or not 0<=p<=1:
                raise ValueError("fake_probability must be finite in [0,1]")
            # Predictions may use a declared threshold other than .5; no forced consistency.
            if row.prediction=="0":
                if row.start_s.strip() or row.end_s.strip():
                    raise ValueError("predicted-real intervals must both be blank")
            else:
                a,b=float(row.start_s),float(row.end_s)
                if not np.isfinite([a,b]).all() or not 0<=a<b<=catalog.loc[row.clip_id,"duration_s"]:
                    raise ValueError("require 0 <= start_s < end_s <= duration")
            if row.cue not in CUES:
                raise ValueError(f"cue must be one of {sorted(CUES)}")
            if len(row.comment.split())>30:
                raise ValueError("comment exceeds 30 whitespace-delimited words")
        except (ValueError,TypeError) as exc:
            errors.append(f"{row.clip_id}: {exc}")
    if errors:
        raise ValueError("\n".join(errors))
    df["prediction"]=df.prediction.astype(int)
    df["fake_probability"]=df.fake_probability.astype(float)
    return df.set_index("clip_id").loc[catalog.index]


def tiou(a,b,c,d):
    intersection=max(0,min(b,d)-max(a,c))
    return intersection/(max(b,d)-min(a,c))


def score_submission(submission,catalog_path,key_path,out,review_path=None):
    sub=validate_submission(submission,catalog_path)
    catalog=load_catalog(catalog_path)
    key=pd.read_csv(key_path,keep_default_na=False,dtype=str)
    if set(["clip_id","label","start_s","end_s","cue"])-set(key.columns):
        raise ValueError("Answer key missing columns")
    if key.clip_id.duplicated().any() or set(key.clip_id)!=set(sub.index):
        raise ValueError("Answer key must match catalog exactly")
    key=key.set_index("clip_id").loc[sub.index]
    if not set(key.label)<={"real","fake"} or not set(key.cue)<=CUES:
        raise ValueError("Invalid answer-key labels/cues")
    review={}
    if review_path:
        r=pd.read_csv(review_path,keep_default_na=False)
        if not {"clip_id","credit","reason"}<=set(r) or r.clip_id.duplicated().any() or not set(r.clip_id)<=set(sub.index):
            raise ValueError("Manual review requires unique known clip_id, credit, reason")
        for row in r.itertuples():
            credit=float(row.credit)
            if not np.isfinite(credit) or not 0<=credit<=1 or not str(row.reason).strip():
                raise ValueError("Manual credit requires finite [0,1] and a written reason")
            review[row.clip_id]=credit
    details=[]
    for clip,row in sub.iterrows():
        truth=key.loc[clip]
        if truth.label=="real":
            if truth.start_s or truth.end_s:
                raise ValueError("Real key interval must be blank")
            loc=float(row.prediction==0)
        else:
            a,b=float(truth.start_s),float(truth.end_s)
            if not np.isfinite([a,b]).all() or not 0<=a<b<=catalog.loc[clip,"duration_s"]:
                raise ValueError("Invalid hidden fake interval")
            loc=tiou(float(row.start_s),float(row.end_s),a,b) if row.prediction==1 else 0.0
        agreement=float(row.cue==truth.cue)
        details.append({"clip_id":clip,"localization":loc,"cue_credit":max(agreement,review.get(clip,0.0))})
    y=key.label.map({"real":0,"fake":1})
    per_clip=pd.DataFrame(details)
    f1=float(f1_score(y,sub.prediction,labels=[0,1],average="macro",zero_division=0))
    result={"macro_f1":f1,"auroc":float(roc_auc_score(y,sub.fake_probability)) if y.nunique()==2 else None,
            "classification_points":70*f1,"localization_points":20*float(per_clip.localization.mean()),
            "interpretation_points":10*float(per_clip.cue_credit.mean()),"n":len(sub),
            "manual_review_applied":bool(review_path),"status":"scored"}
    result["total"]=sum(result[k] for k in ["classification_points","localization_points","interpretation_points"])
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    write_json(out/"score.json",result)
    per_clip.to_csv(out/"per_clip.csv",index=False)
    return result
