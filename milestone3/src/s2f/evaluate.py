"""Clip-level train-only preprocessing and paired baseline comparisons."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from .common import write_json, sha256
from .features import CONTEXT


def records(features, mode):
    rows = []
    for row in features.to_dict("records"):
        # Never include identity, label, filename, generator or annotation provenance.
        result = {k:float(v) for k,v in row.items() if k not in ["clip_id"]+CONTEXT and not k.startswith("ctx_")}
        if mode == "context":
            numeric = {k:float(v) for k,v in row.items() if k.startswith("ctx_")}
            if not numeric or not all(k in row for k in CONTEXT):
                raise ValueError("Context model requires annotated context features")
            result.update(numeric)
            # Explicit feature interactions condition trajectories on each neighboring phoneme.
            # DictVectorizer learns its vocabulary on training clips only; unseen keys ignored.
            for key in CONTEXT:
                for name,value in numeric.items():
                    result[f"{key}={row[key]}::{name}"] = value
        if not all(np.isfinite(list(result.values()))):
            raise ValueError("Non-finite model features")
        rows.append(result)
    return rows


def metrics(y, probability, threshold=.5):
    y = np.asarray(y, int)
    p = np.asarray(probability, float)
    if len(y)==0 or len(y)!=len(p) or not np.isfinite(p).all() or ((p<0)|(p>1)).any():
        raise ValueError("Nonempty aligned labels and finite probabilities in [0,1] required")
    prediction = (p>=threshold).astype(int)
    return {"n":len(y), "macro_f1":float(f1_score(y,prediction,labels=[0,1],average="macro",zero_division=0)),
            "auroc":float(roc_auc_score(y,p)) if len(set(y))==2 else None,
            "confusion_matrix":confusion_matrix(y,prediction,labels=[0,1]).tolist(),
            "threshold":threshold}


def evaluate(manifest, features, out, protocol="speaker", seed=5230, threshold=.5, evidence="pilot"):
    if protocol not in {"clip","speaker","generator","speaker_generator"}:
        raise ValueError("Unknown evaluation protocol")
    for col in ["source_id"] + (["speaker_id"] if "speaker" in protocol else []):
        if (manifest.groupby(col).split.nunique()>1).any():
            raise ValueError(f"Split leakage through {col}")
    if "generator" in protocol and (manifest[manifest.label.eq("fake")].groupby("generator").split.nunique()>1).any():
        raise ValueError("Split leakage through fake generator")
    if not np.isfinite(threshold) or not 0<=threshold<=1:
        raise ValueError("Threshold must be finite in [0,1]")
    if evidence not in {"pilot", "synthetic_smoke_test"}:
        raise ValueError("Evidence must be pilot or synthetic_smoke_test")
    if features.clip_id.duplicated().any() or set(features.clip_id)!=set(manifest.clip_id):
        raise ValueError("Features must contain exactly one row for every manifest clip")
    table = manifest.merge(features, on="clip_id", validate="one_to_one", suffixes=("_manifest", ""))
    train = table.split.eq("train")
    test = table.split.eq("test")
    if not train.any() or not test.any() or table.loc[train,"label"].nunique()!=2:
        raise ValueError("Need nonempty train/test and both classes in training")
    if "evidence_type" in manifest and manifest.evidence_type.str.contains("synthetic").any():
        evidence = "synthetic_smoke_test"
    out = Path(out)
    for folder in ["metrics","predictions","models","figures"]:
        (out/folder).mkdir(parents=True,exist_ok=True)
    modes = ["context_free"] + (["context"] if any(c.startswith("ctx_") for c in features) else [])
    result_rows = []
    for mode in modes:
        x = features.set_index("clip_id").loc[table.clip_id].reset_index()
        data = records(x, mode)
        clf = Pipeline([("vectorize",DictVectorizer(sparse=False)),("scale",StandardScaler()),
                        ("classifier",LogisticRegression(C=1.0,class_weight="balanced",max_iter=2000,random_state=seed))])
        y = table.label.map({"real":0,"fake":1}).to_numpy()
        clf.fit([r for r,t in zip(data,train) if t], y[train])
        bundle = {"pipeline":clf,"mode":mode,"threshold":threshold,"seed":seed,
                  "protocol":protocol,"evidence":evidence,
                  "training_clip_ids":table.loc[train,"clip_id"].tolist(),
                  "training_source_ids":table.loc[train,"source_id"].unique().tolist(),
                  "training_speaker_ids":table.loc[train,"speaker_id"].unique().tolist(),
                  "training_generators":table.loc[train & table.label.eq("fake"),"generator"].unique().tolist()}
        joblib.dump(bundle,out/"models"/f"{mode}.joblib")
        for split in ["val","test"]:
            mask = table.split.eq(split)
            if not mask.any():
                continue
            p = clf.predict_proba([r for r,t in zip(data,mask) if t])[:,1]
            score = metrics(y[mask],p,threshold)
            score.update({"model":mode,"split":split,"protocol":protocol,"seed":seed,"evidence":evidence,
                          "n_train":int(train.sum()),"n_train_speakers":int(table.loc[train,"speaker_id"].nunique()),
                          "n_eval_speakers":int(table.loc[mask,"speaker_id"].nunique())})
            write_json(out/"metrics"/f"{mode}_{split}.json",score)
            pred = table.loc[mask,["clip_id","label","speaker_id","generator","source_id"]].copy()
            pred["fake_probability"] = p
            pred["prediction"] = (p>=threshold).astype(int)
            pred.to_csv(out/"predictions"/f"{mode}_{split}.csv",index=False)
            fig,ax=plt.subplots(figsize=(4.8,4.2))
            ConfusionMatrixDisplay(np.array(score["confusion_matrix"]),display_labels=["real","fake"]).plot(ax=ax,colorbar=False)
            ax.set_title(f"{mode} / {split}\n{evidence}")
            fig.tight_layout()
            fig.savefig(out/"figures"/f"{mode}_{split}_confusion.png",dpi=160)
            plt.close(fig)
            result_rows.append({k:v for k,v in score.items() if k!="confusion_matrix"})
    result = pd.DataFrame(result_rows)
    result.to_csv(out/"metrics"/"results.csv",index=False)
    manifest.to_csv(out/"metrics"/"validated_manifest.csv",index=False)
    features.to_csv(out/"metrics"/"model_features.csv",index=False)
    write_json(out/"metrics"/"run.json",{"protocol":protocol,"seed":seed,"threshold":threshold,
               "evidence":evidence,"features_sha256":sha256(out/"metrics"/"model_features.csv"),
               "manifest_sha256":sha256(out/"metrics"/"validated_manifest.csv"),
               "note":"Fixed C=1.0; no test-set tuning. Validation is diagnostic only. Pilot is not generalization evidence."})
    return result


def predict(model_path, features):
    # joblib is executable serialization: load only your own/trusted trained models.
    bundle=joblib.load(model_path)
    p=bundle["pipeline"].predict_proba(records(features,bundle["mode"]))[:,1]
    result=features[["clip_id"]].copy()
    result["fake_probability"]=p
    result["prediction"]=(p>=bundle["threshold"]).astype(int)
    result["model_evidence"]=bundle["evidence"]
    return result


def evaluate_frozen(model_path, manifest, features, out):
    """Evaluate held-out variants without refitting or changing the threshold."""
    bundle=joblib.load(model_path)
    if not manifest.split.eq("test").all():
        raise ValueError("Frozen robustness evaluation accepts test clips only")
    for col,key in [("clip_id","training_clip_ids"),("source_id","training_source_ids")]:
        if set(manifest[col]) & set(bundle[key]):
            raise ValueError(f"Robustness leakage through {col}")
    if "speaker" in bundle["protocol"] and set(manifest.speaker_id)&set(bundle["training_speaker_ids"]):
        raise ValueError("Robustness speaker leakage")
    if "generator" in bundle["protocol"] and set(manifest.loc[manifest.label.eq("fake"),"generator"])&set(bundle["training_generators"]):
        raise ValueError("Robustness generator leakage")
    if features.clip_id.duplicated().any() or set(features.clip_id)!=set(manifest.clip_id):
        raise ValueError("Frozen feature IDs must match manifest")
    p=predict(model_path,features)
    joined=manifest.merge(p,on="clip_id",validate="one_to_one")
    rows=[]
    for condition,g in joined.groupby("compression"):
        result=metrics(g.label.map({"real":0,"fake":1}),g.fake_probability,bundle["threshold"])
        rows.append({"condition":condition,"model":bundle["mode"],"evidence":bundle["evidence"],**result})
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    joined.to_csv(out/"frozen_predictions.csv",index=False)
    pd.DataFrame([{k:v for k,v in row.items() if k!="confusion_matrix"} for row in rows]).to_csv(out/"robustness_results.csv",index=False)
    write_json(out/"robustness_metrics.json",rows)
    return rows
