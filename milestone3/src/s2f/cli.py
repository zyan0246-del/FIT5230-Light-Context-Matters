import argparse
import json
from pathlib import Path
import pandas as pd
from .common import write_json


def main():
    parser=argparse.ArgumentParser(description="FIT5230 Context Matters M3")
    commands=parser.add_subparsers(dest="command",required=True)
    for name in ["smoke","public-demo"]:
        p=commands.add_parser(name); p.add_argument("--root",default=".")
    p=commands.add_parser("validate-manifest")
    p.add_argument("manifest"); p.add_argument("--root",default=".")
    p.add_argument("--protocol",choices=["speaker","generator","speaker_generator","clip"],default="speaker")
    p=commands.add_parser("preprocess")
    p.add_argument("video"); p.add_argument("--clip-id",required=True)
    p.add_argument("--model",default="models/face_landmarker.task"); p.add_argument("--out",default="outputs/pilot")
    p.add_argument("--stride",type=int,default=1); p.add_argument("--window",type=int,default=5)
    p=commands.add_parser("evaluate")
    p.add_argument("manifest"); p.add_argument("--features",required=True); p.add_argument("--intervals")
    p.add_argument("--root",default="."); p.add_argument("--out",default="outputs/pilot")
    p.add_argument("--protocol",choices=["speaker","generator","speaker_generator","clip"],default="speaker")
    p.add_argument("--config",default="configs/default.json")
    p=commands.add_parser("predict")
    p.add_argument("model"); p.add_argument("feature_table"); p.add_argument("--out",required=True)
    p=commands.add_parser("validate-submission")
    p.add_argument("submission"); p.add_argument("catalog")
    p=commands.add_parser("score")
    p.add_argument("submission"); p.add_argument("catalog"); p.add_argument("key")
    p.add_argument("--out",default="outputs/challenge"); p.add_argument("--review")
    p=commands.add_parser("variants")
    p.add_argument("manifest"); p.add_argument("--root",default="."); p.add_argument("--out",default="data/variants")
    p=commands.add_parser("compare-pairs")
    p.add_argument("pairs"); p.add_argument("predictions"); p.add_argument("--out",required=True)
    p.add_argument("--alpha",type=float)
    p=commands.add_parser("extract-manifest")
    p.add_argument("manifest"); p.add_argument("--root",default="."); p.add_argument("--out",default="outputs/pilot")
    p.add_argument("--model",default="models/face_landmarker.task")
    p=commands.add_parser("feature-table")
    p.add_argument("manifest"); p.add_argument("features"); p.add_argument("--intervals"); p.add_argument("--out",required=True)
    p=commands.add_parser("evaluate-frozen")
    p.add_argument("model"); p.add_argument("manifest"); p.add_argument("feature_table")
    p.add_argument("--root",default="."); p.add_argument("--out",default="outputs/robustness")
    p=commands.add_parser("analyse-colour-pair")
    p.add_argument("control"); p.add_argument("modified")
    p.add_argument("--out",default="outputs/m3_work/chromatic_pair")
    p.add_argument("--expected-hz",type=float,default=1.2)
    args=parser.parse_args()
    try:
        if args.command=="smoke":
            from .demo import synthetic_demo
            synthetic_demo(args.root)
        elif args.command=="public-demo":
            from .demo import public_demo
            public_demo(args.root)
        elif args.command=="validate-manifest":
            from .manifest import validate_manifest
            print(validate_manifest(args.manifest,args.root,args.protocol).to_string(index=False))
        elif args.command=="preprocess":
            from .preprocess import preprocess
            print(json.dumps(preprocess(args.video,args.clip_id,args.model,args.out,args.stride,args.window),indent=2))
        elif args.command=="evaluate":
            from .manifest import validate_manifest
            from .features import build_features,read_intervals
            from .evaluate import evaluate
            config=json.loads(Path(args.config).read_text())
            m=validate_manifest(args.manifest,args.root,args.protocol)
            f=build_features(m,args.features,read_intervals(args.intervals) if args.intervals else None,
                             config["min_coverage"],config["context_window_s"],config["min_segment_frames"])
            print(evaluate(m,f,args.out,args.protocol,config["seed"],config["decision_threshold"]).to_string(index=False))
            write_json(Path(args.out)/"metrics"/"config.json",config)
        elif args.command=="predict":
            from .evaluate import predict
            predict(args.model,pd.read_csv(args.feature_table)).to_csv(args.out,index=False)
        elif args.command=="validate-submission":
            from .challenge import validate_submission
            print(f"Valid: {len(validate_submission(args.submission,args.catalog))} clips")
        elif args.command=="score":
            from .challenge import score_submission
            print(json.dumps(score_submission(args.submission,args.catalog,args.key,args.out,args.review),indent=2))
        elif args.command=="variants":
            from .manifest import validate_manifest
            from .robustness import make_variants
            print(make_variants(validate_manifest(args.manifest,args.root,protocol="clip"),args.root,args.out).to_string(index=False))
        elif args.command=="compare-pairs":
            from .interaction import compare_pairs
            print(compare_pairs(args.pairs,args.predictions,args.out,args.alpha).to_string(index=False))
        elif args.command=="extract-manifest":
            from .preprocess import preprocess
            from .manifest import validate_manifest
            m=validate_manifest(args.manifest,args.root,protocol="clip")
            results=[preprocess(Path(args.root)/r.file_path,r.clip_id,args.model,args.out) for r in m.itertuples()]
            pd.DataFrame(results).to_csv(Path(args.out)/"qc.csv",index=False)
        elif args.command=="feature-table":
            from .features import build_features,read_intervals
            m=pd.read_csv(args.manifest,keep_default_na=False)
            f=build_features(m,args.features,read_intervals(args.intervals) if args.intervals else None)
            f.to_csv(args.out,index=False)
        elif args.command=="evaluate-frozen":
            from .manifest import validate_manifest
            from .evaluate import evaluate_frozen
            m=validate_manifest(args.manifest,args.root,protocol="clip")
            print(json.dumps(evaluate_frozen(args.model,m,pd.read_csv(args.feature_table),args.out),indent=2))
        elif args.command=="analyse-colour-pair":
            from .chromatic import analyse_pair
            print(json.dumps(analyse_pair(args.control,args.modified,args.out,args.expected_hz),indent=2))
    except (ValueError,FileNotFoundError) as exc:
        parser.exit(2,f"ERROR: {exc}\n")


if __name__=="__main__":
    main()
