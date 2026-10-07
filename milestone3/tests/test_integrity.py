from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import pandas as pd
import pytest
from s2f.manifest import validate_manifest,REQUIRED
from s2f.preprocess import geometry,add_dynamics
from s2f.features import read_intervals,build_features
from s2f.evaluate import metrics,records
from s2f.challenge import validate_submission,score_submission
from s2f.interaction import compare_pairs


def manifest(tmp_path):
    rows=[]
    for i,(label,split) in enumerate([("real","train"),("fake","train"),("real","test"),("fake","test")]):
        file=tmp_path/f"video{i}.mp4"; file.write_bytes(f"content{i}".encode())
        row=dict.fromkeys(REQUIRED,"unknown")
        row.update(clip_id=f"c{i}",file_path=file.name,label=label,split=split,speaker_id=f"s{i}",
                   source_id=f"source{i}",generator=f"gen{i}")
        rows.append(row)
    return pd.DataFrame(rows)


@pytest.mark.parametrize("problem",["missing","duplicate_id","duplicate_contents","source","speaker","generator","bad_label","bad_id"])
def test_manifest_rejects_leakage(tmp_path,problem):
    df=manifest(tmp_path)
    if problem=="missing": (tmp_path/"video0.mp4").unlink()
    elif problem=="duplicate_id": df.loc[2,"clip_id"]="c0"
    elif problem=="duplicate_contents": (tmp_path/"video2.mp4").write_bytes(b"content0")
    elif problem=="source": df.loc[2,"source_id"]="source0"
    elif problem=="speaker": df.loc[2,"speaker_id"]="s0"
    elif problem=="generator": df.loc[3,"generator"]="gen1"
    elif problem=="bad_label": df.loc[0,"label"]="maybe"
    elif problem=="bad_id": df.loc[0,"clip_id"]="../escape"
    path=tmp_path/"m.csv";df.to_csv(path,index=False)
    with pytest.raises(ValueError): validate_manifest(path,tmp_path,"speaker_generator")


def test_valid_manifest(tmp_path):
    df=manifest(tmp_path);df.to_csv(tmp_path/"m.csv",index=False)
    assert len(validate_manifest(tmp_path/"m.csv",tmp_path,"speaker_generator"))==4


def test_geometry_aspect_ratio():
    points=[SimpleNamespace(x=0.,y=0.) for _ in range(468)]
    points[263].x=.2
    points[14].y=.1
    assert geometry(points,200,100)["inner_aperture"]==pytest.approx(.25)


def frame():
    t=np.arange(100)/30
    return pd.DataFrame({"time_s":t,"detected":True,"inner_aperture":t**3,
                         "outer_aperture":t**3+.1,"lip_width":.5+t*.01})


def test_derivatives_seconds_and_gaps():
    f=frame();f.loc[35:45,"detected"]=False
    f.loc[35:45,["inner_aperture","outer_aperture","lip_width"]]=np.nan
    r=add_dynamics(f,1)
    assert r.loc[20,"aperture_jerk"]==pytest.approx(6,abs=1e-6)
    assert r.loc[35:45,"inner_aperture_smooth"].isna().all()
    assert r.loc[32:48,"aperture_jerk"].isna().all()


@pytest.mark.parametrize("start,end",[(-1,1),(1,1),(2,1),(float("nan"),2),(0,float("inf"))])
def test_interval_invalid(tmp_path,start,end):
    pd.DataFrame([dict(clip_id="c1",start_s=start,end_s=end,previous_phoneme="a",central_phoneme="b",
                       next_phoneme="u",annotation_source="manual")]).to_csv(tmp_path/"i.csv",index=False)
    with pytest.raises(ValueError):read_intervals(tmp_path/"i.csv")


def test_context_changes_representation():
    a=pd.DataFrame([dict(clip_id="x",global_mean=.1,ctx_before=.2,previous_phoneme="a",central_phoneme="b",next_phoneme="i")])
    b=a.copy();b["next_phoneme"]="u"
    assert records(a,"context_free")==records(b,"context_free")
    assert records(a,"context")!=records(b,"context")


def test_one_class_auroc_is_null():
    assert metrics([0,0],[.1,.2])["auroc"] is None


@pytest.mark.parametrize("field,value",[("fake_probability","nan"),("fake_probability","1.1"),("prediction","2"),
                                     ("start_s",""),("end_s","9"),("cue","nonsense"),("comment","word "*31)])
def test_submission_rejects(tmp_path,field,value):
    df=pd.read_csv("challenge/dummy_submission.csv",keep_default_na=False,dtype=str)
    df.loc[1,field]=value
    df.to_csv(tmp_path/"s.csv",index=False)
    with pytest.raises(ValueError):validate_submission(tmp_path/"s.csv","challenge/dummy_catalog.csv")


def test_challenge_known_score(tmp_path):
    score=score_submission("challenge/dummy_submission.csv","challenge/dummy_catalog.csv","challenge/dummy_answer_key.csv",tmp_path)
    assert score["total"]==pytest.approx(98)


def test_missing_rppg_stays_missing(tmp_path):
    pairs=pd.DataFrame([dict(pair_id="p1",variant=v,clip_id=f"c{i}",rppg_fake_probability="") for i,v in enumerate(["original","modified"])])
    pairs.to_csv(tmp_path/"p.csv",index=False)
    pd.DataFrame({"clip_id":["c0","c1"],"fake_probability":[.8,.6]}).to_csv(tmp_path/"s.csv",index=False)
    result=compare_pairs(tmp_path/"p.csv",tmp_path/"s.csv",tmp_path/"out.csv",.5)
    assert result.fused_fake_probability.isna().all()
    assert result.iloc[1].lip_fake_probability_delta_from_original==pytest.approx(-.2)
