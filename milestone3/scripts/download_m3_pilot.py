"""Download and standardise the small M3 public pilot.

Raw and processed videos are ignored by Git. The script records URLs, hashes,
Wikimedia license metadata and the exact FFmpeg command in the manifest.
"""
from pathlib import Path
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen
import hashlib
import json
import shutil
import subprocess
import sys
import time
from urllib.error import HTTPError, URLError

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "data" / "m3_pilot_sources.json"
RAW = ROOT / "data" / "raw" / "m3_pilot"
PROCESSED = ROOT / "data" / "processed" / "m3_pilot"
MANIFEST = ROOT / "data" / "m3_pilot_manifest.csv"
PROVENANCE = ROOT / "data" / "m3_pilot_provenance.json"
HF_BASE = "https://huggingface.co/datasets/luchaoqi/TalkingHeadBench/resolve/main/"
COMMONS_REDIRECT = "https://commons.wikimedia.org/wiki/Special:Redirect/file/"
COMMONS_API = "https://commons.wikimedia.org/w/api.php"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url, target):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 0:
        return
    temporary = target.with_suffix(target.suffix + ".part")
    for attempt in range(6):
        try:
            request = Request(url, headers={"User-Agent": "FIT5230-course-research/1.0"})
            with urlopen(request, timeout=120) as source, temporary.open("wb") as sink:
                shutil.copyfileobj(source, sink)
            temporary.replace(target)
            return
        except (HTTPError, URLError, TimeoutError):
            temporary.unlink(missing_ok=True)
            if attempt == 5:
                raise
            time.sleep(3 * (2 ** attempt))


def commons_metadata(filename):
    params = {
        "action": "query",
        "format": "json",
        "prop": "videoinfo",
        "viprop": "url|derivatives|extmetadata",
        "titles": "File:" + filename,
    }
    for attempt in range(6):
        try:
            request = Request(COMMONS_API + "?" + urlencode(params), headers={"User-Agent": "FIT5230-course-research/1.0"})
            with urlopen(request, timeout=60) as response:
                payload = json.load(response)
            break
        except (HTTPError, URLError, TimeoutError):
            if attempt == 5:
                raise
            time.sleep(3 * (2 ** attempt))
    page = next(iter(payload["query"]["pages"].values()))
    info = page["videoinfo"][0]
    ext = info.get("extmetadata", {})
    derivatives = info.get("derivatives", [])
    preferred = next((d for d in derivatives if d.get("transcodekey") == "240p.vp9.webm"), None)
    preferred = preferred or next((d for d in derivatives if d.get("height", 9999) <= 360), None)
    return {
        "canonical_url": info.get("descriptionurl"),
        "download_url": info.get("url"),
        "license_short_name": ext.get("LicenseShortName", {}).get("value"),
        "license_url": ext.get("LicenseUrl", {}).get("value"),
        "artist": ext.get("Artist", {}).get("value"),
        "credit": ext.get("Credit", {}).get("value"),
        "selected_derivative": preferred,
    }


def transcode(source, target, start_s=0):
    target.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", str(start_s), "-i", str(source), "-t", "12",
        "-vf", "fps=25,scale=512:512:force_original_aspect_ratio=decrease,pad=512:512:(ow-iw)/2:(oh-ih)/2",
        "-c:v", "libx264", "-crf", "23", "-preset", "veryfast", "-pix_fmt", "yuv420p",
        "-ac", "1", "-ar", "16000", "-c:a", "aac", "-map_metadata", "-1", str(target),
    ]
    subprocess.run(command, check=True)
    # Record repository-relative paths so the provenance file is portable and
    # does not disclose the machine/user directory that produced the pilot.
    portable = command.copy()
    portable[portable.index(str(source))] = source.relative_to(ROOT).as_posix()
    portable[portable.index(str(target))] = target.relative_to(ROOT).as_posix()
    return portable


def main():
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is required")
    config = json.loads(CONFIG.read_text())
    rows, provenance = [], []
    for item in config["real"]:
        metadata = commons_metadata(item["filename"])
        derivative = metadata.get("selected_derivative") or {}
        url = derivative.get("src") or metadata["download_url"] or (COMMONS_REDIRECT + quote(item["filename"]))
        suffix = Path(url.split("?")[0]).suffix or ".webm"
        raw = RAW / "real" / f"{item['id']}{suffix}"
        processed = PROCESSED / f"{item['id']}.mp4"
        print("real", item["id"], flush=True)
        download(url, raw)
        command = transcode(raw, processed, item["start_s"])
        rows.append({
            "clip_id": item["id"], "file_path": processed.relative_to(ROOT).as_posix(),
            "label": "real", "speaker_id": item["speaker"], "generator": "none",
            "transcript": "unknown", "central_phoneme": "unknown",
            "previous_phoneme": "unknown", "next_phoneme": "unknown",
            "compression": "standardized_h264_crf23", "split": item["split"],
            "source_id": item["id"],
        })
        provenance.append({"clip_id": item["id"], "source": "Wikimedia Commons/Wikitongues",
                           "metadata": metadata, "raw_sha256": sha256(raw),
                           "processed_sha256": sha256(processed), "ffmpeg": command})
    for item in config["fake"]:
        url = HF_BASE + quote(item["path"], safe="/")
        raw = RAW / "fake" / f"{item['id']}.mp4"
        processed = PROCESSED / f"{item['id']}.mp4"
        print("fake", item["id"], flush=True)
        download(url, raw)
        command = transcode(raw, processed, 0)
        rows.append({
            "clip_id": item["id"], "file_path": processed.relative_to(ROOT).as_posix(),
            "label": "fake", "speaker_id": item["speaker"], "generator": item["generator"],
            "transcript": "unknown", "central_phoneme": "unknown",
            "previous_phoneme": "unknown", "next_phoneme": "unknown",
            "compression": "standardized_h264_crf23", "split": item["split"],
            "source_id": item["id"],
        })
        provenance.append({"clip_id": item["id"], "source": "TalkingHeadBench",
                           "source_url": url, "dataset_card": "https://huggingface.co/datasets/luchaoqi/TalkingHeadBench",
                           "upstream_licenses_must_be_respected": True, "raw_sha256": sha256(raw),
                           "processed_sha256": sha256(processed), "ffmpeg": command})
    frame = pd.DataFrame(rows).sort_values(["split", "label", "clip_id"])
    frame.to_csv(MANIFEST, index=False)
    PROVENANCE.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(frame.groupby(["split", "label"]).size())
    print("manifest", MANIFEST)


if __name__ == "__main__":
    main()
