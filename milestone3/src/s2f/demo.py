"""Deterministic synthetic *feature* smoke test, not a deepfake dataset."""
from pathlib import Path
import importlib.metadata
import platform
import sys
import shutil
import urllib.request
import subprocess
import numpy as np
import pandas as pd
import imageio_ffmpeg
from .common import write_json, sha256
from .preprocess import add_dynamics, plot_trajectory, preprocess
from .manifest import validate_manifest
from .features import read_intervals, build_features
from .evaluate import evaluate
from .challenge import score_submission

MODEL_URL="https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
REAL_URL="https://commons.wikimedia.org/wiki/Special:Redirect/file/WIKITONGUES-_Simon_speaking_Cumbrian.webm"
FAKE_URL="https://huggingface.co/datasets/luchaoqi/TalkingHeadBench/resolve/main/fake/Hallo/test/00023--7lHr2ZNlULA_4--Hallo.mp4?download=true"


def download(url,destination):
    destination=Path(destination)
    destination.parent.mkdir(parents=True,exist_ok=True)
    if not destination.exists():
        temporary=destination.with_suffix(destination.suffix+".part")
        request=urllib.request.Request(url,headers={"User-Agent":"FIT5230-M2-Defensive-Research/0.1"})
        with urllib.request.urlopen(request,timeout=90) as response, temporary.open("wb") as stream:
            shutil.copyfileobj(response,stream)
        temporary.replace(destination)
    if destination.stat().st_size==0:
        raise ValueError(f"Empty download: {destination}")
    return destination


def environment():
    names=["numpy","pandas","scipy","scikit-learn","matplotlib","mediapipe","opencv-contrib-python","nbformat","nbclient"]
    return {"python":sys.version,"platform":platform.platform(),
            "packages":{name:importlib.metadata.version(name) for name in names}}


def synthetic_demo(root="."):
    root=Path(root)
    raw=root/"data"/"synthetic"
    output=root/"outputs"/"smoke"
    for path in [raw,output/"features",output/"figures"]:
        path.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(5230)
    manifest=[]; annotations=[]
    # 16 artificial speaker IDs, 2 clips each, both classes per artificial speaker.
    # IDs provide split-mechanics coverage only; they are NOT real independent speakers.
    for speaker in range(16):
        split="train" if speaker<10 else "val" if speaker<12 else "test"
        for label in ["real","fake"]:
            clip=f"toy_{speaker:02d}_{label}"
            t=np.arange(120)/30
            context="iy" if speaker%2==0 else "uw"
            phase=rng.uniform(-.4,.4)
            x=.13+.045*np.sin(2*np.pi*1.4*t+phase)+rng.normal(0,.003,len(t))
            # Arbitrary toy signal differences exercise numerical code, not physiology.
            if label=="fake":
                x+=.015*np.sin(2*np.pi*6*t)
            width=.72+.05*np.sin(2*np.pi*.8*t+phase)
            width+=(1 if context=="iy" else -1)*.025*np.tanh((t-1.8)*6)
            df=pd.DataFrame({"frame_index":np.arange(len(t)),"time_s":t,"detected":True,
                             "inner_aperture":x,"outer_aperture":x+.045,"lip_width":width})
            df=add_dynamics(df,5)
            df.to_csv(raw/f"{clip}.csv",index=False)
            df.to_csv(output/"features"/f"{clip}.csv",index=False)
            manifest.append({"clip_id":clip,"file_path":f"data/synthetic/{clip}.csv","label":label,
                             "speaker_id":f"artificial_{speaker:02d}","generator":"toy_math" if label=="fake" else "none",
                             "transcript":"synthetic fixture; no speech","previous_phoneme":"sil","central_phoneme":"b",
                             "next_phoneme":context,"compression":"none_features_only","split":split,"source_id":f"toy_source_{speaker:02d}",
                             "evidence_type":"synthetic_feature_fixture"})
            annotations.append({"clip_id":clip,"start_s":1.5,"end_s":1.9,"previous_phoneme":"sil",
                                "central_phoneme":"b","next_phoneme":context,"annotation_source":"synthetic_fixture_not_alignment"})
    pd.DataFrame(manifest).to_csv(raw/"manifest.csv",index=False)
    pd.DataFrame(annotations).to_csv(raw/"intervals.csv",index=False)
    m=validate_manifest(raw/"manifest.csv",root)
    features=build_features(m,output/"features",read_intervals(raw/"intervals.csv"),context_window=.3)
    results=evaluate(m,features,output,evidence="synthetic_smoke_test")
    plot_trajectory(pd.read_csv(raw/"toy_12_fake.csv"),output/"figures"/"toy_trajectory.png","Synthetic trajectory fixture - not measured facial motion")
    write_json(output/"environment.json",environment())
    # Preserve M1 70/20/10 rules on a clearly separate dummy answer key.
    score_submission(root/"challenge"/"dummy_submission.csv",root/"challenge"/"dummy_catalog.csv",
                     root/"challenge"/"dummy_answer_key.csv",output/"challenge")
    print(results.to_string(index=False))
    print("SYNTHETIC SMOKE TEST ONLY: no biological, deepfake-detection or generalization claim.")
    return results


def public_demo(root="."):
    root=Path(root); raw=root/"data"/"raw"/"public_demo"; out=root/"outputs"/"public_demo"
    raw.mkdir(parents=True,exist_ok=True)
    model=download(MODEL_URL,root/"models"/"face_landmarker.task")
    real_source=download(REAL_URL,raw/"wikitongues_real.webm")
    fake=download(FAKE_URL,raw/"hallo_fake.mp4")
    real=raw/"wikitongues_real_12s.mp4"
    if not real.exists():
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),"-hide_banner","-loglevel","error","-y","-ss","2","-i",str(real_source),
                        "-t","12","-c:v","libx264","-crf","23","-c:a","aac",str(real)],check=True)
    rows=[]
    for clip,path in [("public_real",real),("public_hallo",fake)]:
        rows.append(preprocess(path,clip,model,out))
    pd.DataFrame(rows).to_csv(out/"qc.csv",index=False)
    write_json(out/"environment.json",environment())
    write_json(out/"provenance.json",{"real_source":REAL_URL,"fake_source":FAKE_URL,"model_source":MODEL_URL,
              "real_sha256":sha256(real),"fake_sha256":sha256(fake),
              "interpretation":"Unmatched speakers and speech. Preprocessing smoke test only. No classifier accuracy."})
    print(pd.DataFrame(rows)[["clip_id","sampled_frames","coverage","status"]].to_string(index=False))
    return rows
