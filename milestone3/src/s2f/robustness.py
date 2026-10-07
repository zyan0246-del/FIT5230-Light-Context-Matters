"""FFmpeg variants: retain source IDs/splits; score with frozen clean models."""
from pathlib import Path
import subprocess
import imageio_ffmpeg
import pandas as pd
from .preprocess import get_video_metadata
from .common import write_json, sha256

VARIANTS={"h264_crf32":[],"resize_256":["-vf","scale=-2:256"],"fps_15":["-vf","fps=15"]}


def make_variants(manifest,root,out,only_test=True):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    rows=[]; logs=[]
    for row in manifest.to_dict("records"):
        if only_test and row["split"]!="test":
            continue
        source=Path(root)/row["file_path"]
        for name,filters in VARIANTS.items():
            target=out/f"{row['clip_id']}__{name}.mp4"
            command=[imageio_ffmpeg.get_ffmpeg_exe(),"-hide_banner","-loglevel","error","-y","-i",str(source),
                     "-map","0:v:0","-map","0:a?",*filters,"-c:v","libx264","-crf","32" if name=="h264_crf32" else "23",
                     "-preset","veryfast","-c:a","aac","-map_metadata","-1",str(target)]
            subprocess.run(command,check=True,capture_output=True)
            record={**row,"clip_id":target.stem,"file_path":target.resolve().relative_to(Path(root).resolve()).as_posix(),
                    "compression":name,"parent_clip_id":row["clip_id"],"sha256":sha256(target)}
            rows.append(record)
            logs.append({"clip_id":target.stem,"variant":name,"metadata":get_video_metadata(target),"command":command})
    if not rows:
        raise ValueError("No eligible clips for robustness variants")
    frame=pd.DataFrame(rows)
    frame.to_csv(out/"manifest.csv",index=False)
    write_json(out/"transcodes.json",logs)
    return frame
