"""Build a small matched real/SadTalker course pilot.

Each public real speaker supplies the still image and audio for exactly one
SadTalker clip.  The real/fake pair shares identity, audio, duration, split and
final codec settings.  This controls the largest confound in the original
cross-collection pilot while remaining small enough for FIT5230 M3.

The script is resumable.  Raw/derived media are ignored by Git; the manifest
and provenance JSON are suitable for audit after private paths are removed.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from urllib.parse import quote

import pandas as pd

from download_m3_pilot import (
    COMMONS_REDIRECT,
    ROOT,
    commons_metadata,
    download,
    sha256,
)


CONFIG = ROOT / "data" / "m3_pilot_sources.json"
SOURCE_PROVENANCE = ROOT / "data" / "m3_pilot_provenance.json"
RAW_ROOT = ROOT / "data" / "raw" / "matched_sadtalker"
PROCESSED_ROOT = ROOT / "data" / "processed" / "matched_sadtalker"
DEFAULT_MANIFEST = ROOT / "data" / "matched_sadtalker_manifest.csv"
DEFAULT_PROVENANCE = ROOT / "data" / "matched_sadtalker_provenance.json"


def run(command: list[str]) -> None:
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def ffprobe(path: Path) -> dict:
    payload = subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries",
        "format=duration,size:stream=codec_type,codec_name,width,height,r_frame_rate",
        "-of", "json", str(path),
    ], text=True)
    return json.loads(payload)


def decoded_audio_md5(path: Path) -> str:
    output = subprocess.check_output([
        "ffmpeg", "-v", "error", "-i", str(path),
        "-map", "0:a:0", "-f", "md5", "-",
    ], text=True).strip()
    if not output.startswith("MD5="):
        raise RuntimeError(f"Could not hash decoded audio in {path}")
    return output.removeprefix("MD5=")


def validate_pair(real_path: Path, fake_path: Path, duration_s: float) -> tuple[dict, dict, str]:
    real_probe, fake_probe = ffprobe(real_path), ffprobe(fake_path)
    for name, probe in [("real", real_probe), ("fake", fake_probe)]:
        video = next((stream for stream in probe["streams"] if stream["codec_type"] == "video"), None)
        if video is None or (video.get("width"), video.get("height")) != (512, 512):
            raise ValueError(f"{name} output is not 512x512")
        if video.get("r_frame_rate") != "25/1":
            raise ValueError(f"{name} output is not 25 FPS")
        actual_duration = float(probe["format"]["duration"])
        if abs(actual_duration - duration_s) > 0.15:
            raise ValueError(f"{name} duration {actual_duration:.3f}s differs from target")
    real_audio, fake_audio = decoded_audio_md5(real_path), decoded_audio_md5(fake_path)
    if real_audio != fake_audio:
        raise ValueError("Matched real/fake outputs do not have identical decoded audio")
    return real_probe, fake_probe, real_audio


def transcode_real_command(
    raw: Path, target: Path, start_s: float, duration_s: float,
) -> list[str]:
    return [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", str(start_s), "-i", str(raw), "-t", str(duration_s),
        "-vf", "fps=25,scale=512:512:force_original_aspect_ratio=decrease,"
               "pad=512:512:(ow-iw)/2:(oh-ih)/2",
        "-c:v", "libx264", "-crf", "23", "-preset", "veryfast",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-ac", "1", "-ar", "16000",
        "-map_metadata", "-1", str(target),
    ]


def transcode_real(raw: Path, target: Path, start_s: float, duration_s: float) -> list[str]:
    target.parent.mkdir(parents=True, exist_ok=True)
    command = transcode_real_command(raw, target, start_s, duration_s)
    run(command)
    return command


def make_inputs(real_video: Path, portrait: Path, audio: Path, duration_s: float) -> None:
    portrait.parent.mkdir(parents=True, exist_ok=True)
    audio.parent.mkdir(parents=True, exist_ok=True)
    if not portrait.exists():
        run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-ss", "1.0", "-i", str(real_video), "-frames:v", "1", str(portrait),
        ])
    if not audio.exists():
        run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(real_video), "-t", str(duration_s), "-vn",
            "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(audio),
        ])


def generate_fake(
    portrait: Path,
    audio: Path,
    target: Path,
    sadtalker_root: Path,
    sadtalker_python: Path,
    work_root: Path,
) -> tuple[list[str], bool]:
    run_dir = work_root / target.stem
    command = [
        str(sadtalker_python), "inference.py",
        "--driven_audio", str(audio),
        "--source_image", str(portrait),
        "--checkpoint_dir", "checkpoints",
        "--result_dir", str(run_dir),
        "--size", "256", "--batch_size", "1",
        "--preprocess", "full", "--still",
    ]
    if target.exists() and target.stat().st_size > 0:
        return command, False
    run_dir.mkdir(parents=True, exist_ok=True)
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, cwd=sadtalker_root, check=True)
    candidates = sorted(run_dir.glob("*.mp4"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise RuntimeError(f"SadTalker produced no MP4 for {target.stem}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(candidates[-1], target)
    return command, True


def standardise(
    video: Path, audio: Path, target: Path, duration_s: float,
) -> tuple[list[str], bool]:
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(video), "-i", str(audio), "-t", str(duration_s),
        "-map", "0:v:0", "-map", "1:a:0",
        "-vf", "fps=25,scale=512:512:force_original_aspect_ratio=decrease,"
               "pad=512:512:(ow-iw)/2:(oh-ih)/2",
        "-c:v", "libx264", "-crf", "23", "-preset", "veryfast",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-ac", "1", "-ar", "16000",
        "-map_metadata", "-1", str(target),
    ]
    if target.exists() and target.stat().st_size > 0:
        return command, False
    target.parent.mkdir(parents=True, exist_ok=True)
    run(command)
    return command, True


def portable(command: list[str], replacements: dict[Path, str]) -> list[str]:
    result = []
    for token in command:
        replacement = next((name for path, name in replacements.items() if token == str(path)), None)
        result.append(replacement or token)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=12.0)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--provenance", type=Path, default=DEFAULT_PROVENANCE)
    parser.add_argument(
        "--sadtalker-root", type=Path,
        default=Path(os.environ.get("SADTALKER_ROOT", "/root/gpufree-data/generators/SadTalker-cd4c046")),
    )
    parser.add_argument(
        "--sadtalker-python", type=Path,
        default=Path(os.environ.get("SADTALKER_PYTHON", "/root/gpufree-data/sadtalker-env/bin/python")),
    )
    parser.add_argument(
        "--work-root", type=Path,
        default=Path(os.environ.get("SADTALKER_WORK_ROOT", "/root/gpufree-data/matched-sadtalker-work")),
    )
    args = parser.parse_args()

    for executable in ["ffmpeg", "ffprobe"]:
        if not shutil.which(executable):
            raise RuntimeError(f"{executable} is required")
    if not args.sadtalker_python.is_file() or not (args.sadtalker_root / "inference.py").is_file():
        raise FileNotFoundError("Prepared SadTalker environment was not found")

    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    saved_sources = {
        item["clip_id"]: item["metadata"]
        for item in json.loads(SOURCE_PROVENANCE.read_text(encoding="utf-8"))
        if item.get("clip_id", "").startswith("real_") and item.get("metadata")
    }
    rows: list[dict] = []
    provenance: list[dict] = []
    for item in config["real"]:
        clip_id = item["id"].removeprefix("real_")
        pair_id = f"matched_{clip_id}"
        real_id, fake_id = f"matched_real_{clip_id}", f"matched_fake_{clip_id}"
        # Prefer the already-audited M3 provenance.  This avoids making the
        # reproducibility path depend on a live Commons API response while
        # retaining the canonical URL, licence and exact selected derivative.
        metadata = saved_sources.get(item["id"]) or commons_metadata(item["filename"])
        derivative = metadata.get("selected_derivative") or {}
        url = derivative.get("src") or metadata["download_url"] or (
            COMMONS_REDIRECT + quote(item["filename"])
        )
        suffix = Path(url.split("?")[0]).suffix or ".webm"
        raw_source = RAW_ROOT / "source" / f"{clip_id}{suffix}"
        source_silent = RAW_ROOT / "source_silent" / f"{clip_id}.mp4"
        portrait = RAW_ROOT / "portrait" / f"{clip_id}.png"
        audio = RAW_ROOT / "audio" / f"{clip_id}.wav"
        generated_raw = RAW_ROOT / "generated" / f"{clip_id}.mp4"
        real_final = PROCESSED_ROOT / f"{real_id}.mp4"
        fake_final = PROCESSED_ROOT / f"{fake_id}.mp4"

        print(f"=== {pair_id} ===", flush=True)
        download(url, raw_source)
        source_command = transcode_real_command(
            raw_source, source_silent, item["start_s"], args.duration,
        )
        source_executed = False
        # Recreate the source if a previous interrupted run stopped before the
        # shared audio was extracted (including legacy video-only intermediates).
        if not source_silent.exists() or not audio.exists():
            source_command = transcode_real(raw_source, source_silent, item["start_s"], args.duration)
            source_executed = True
        make_inputs(source_silent, portrait, audio, args.duration)
        generation_command, generation_executed = generate_fake(
            portrait, audio, generated_raw, args.sadtalker_root,
            args.sadtalker_python, args.work_root,
        )
        real_command, real_executed = standardise(
            source_silent, audio, real_final, args.duration,
        )
        fake_command, fake_executed = standardise(
            generated_raw, audio, fake_final, args.duration,
        )
        real_probe, fake_probe, audio_pcm_md5 = validate_pair(real_final, fake_final, args.duration)

        for current_id, label, generator, path in [
            (real_id, "real", "none", real_final),
            (fake_id, "fake", "SadTalker_cd4c046", fake_final),
        ]:
            rows.append({
                "clip_id": current_id,
                "file_path": path.relative_to(ROOT).as_posix(),
                "label": label,
                "speaker_id": item["speaker"],
                "generator": generator,
                "transcript": "unknown",
                "central_phoneme": "unknown",
                "previous_phoneme": "unknown",
                "next_phoneme": "unknown",
                "compression": "paired_h264_crf23_512_25fps_aac16k",
                "split": item["split"],
                "source_id": pair_id,
            })

        replacements = {
            raw_source: "<downloaded_public_source>",
            source_silent: "<matched_source_video>",
            portrait: "<source_portrait>",
            audio: "<shared_audio>",
            generated_raw: "<raw_sadtalker_output>",
            real_final: real_final.relative_to(ROOT).as_posix(),
            fake_final: fake_final.relative_to(ROOT).as_posix(),
        }
        provenance.append({
            "pair_id": pair_id,
            "speaker_id": item["speaker"],
            "split": item["split"],
            "public_metadata": metadata,
            "download_url": url,
            "raw_source_sha256": sha256(raw_source),
            "portrait_sha256": sha256(portrait),
            "shared_audio_sha256": sha256(audio),
            "real_sha256": sha256(real_final),
            "fake_sha256": sha256(fake_final),
            "real_probe": real_probe,
            "fake_probe": fake_probe,
            "decoded_audio_pcm_md5": audio_pcm_md5,
            "source_command": portable(source_command, replacements),
            "generation_command": portable(generation_command, replacements),
            "real_standardise_command": portable(real_command, replacements),
            "fake_standardise_command": portable(fake_command, replacements),
            "execution_status_this_run": {
                "source_transcode": "executed" if source_executed else "reused_verified_output",
                "sadtalker_generation": "executed" if generation_executed else "reused_verified_output",
                "real_standardisation": "executed" if real_executed else "reused_verified_output",
                "fake_standardisation": "executed" if fake_executed else "reused_verified_output",
            },
            "model": {
                "name": "SadTalker",
                "commit": "cd4c0465ae0b54a6f85af57f5c65fec9fe23e7f8",
                "checkpoint": "SadTalker_V0.0.2_256.safetensors",
                "preprocess": "full",
                "still": True,
            },
        })
        pd.DataFrame(rows).sort_values(["split", "source_id", "label"]).to_csv(args.manifest, index=False)
        args.provenance.write_text(json.dumps(provenance, indent=2), encoding="utf-8")

    frame = pd.DataFrame(rows).sort_values(["split", "source_id", "label"])
    frame.to_csv(args.manifest, index=False)
    args.provenance.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(frame.groupby(["split", "label"]).size())
    print("manifest", args.manifest)
    print("provenance", args.provenance)


if __name__ == "__main__":
    main()
