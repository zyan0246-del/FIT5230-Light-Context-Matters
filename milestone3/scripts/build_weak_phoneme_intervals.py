"""Create a reproducible *weak* phoneme-context pilot annotation.

Whisper supplies word timestamps. CMUdict supplies an English pronunciation for
each recognised word. Phones are placed uniformly inside a word, so these are
not forced-alignment ground truth. The script deliberately records that
limitation in every row and keeps the raw ASR output for audit.

For the small course pilot we use broad articulatory classes as the categorical
context. Exact ARPAbet phones are retained as extra audit columns.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cmudict
import pandas as pd
import whisper


ROOT = Path(__file__).resolve().parents[1]
WORD_RE = re.compile(r"[A-Za-z']+")
VOWELS = {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"}
STOPS = {"P", "B", "T", "D", "K", "G"}
FRICATIVES = {"F", "V", "TH", "DH", "S", "Z", "SH", "ZH", "HH"}
AFFRICATES = {"CH", "JH"}
NASALS = {"M", "N", "NG"}
LIQUIDS = {"L", "R"}
GLIDES = {"W", "Y"}


def clean_phone(phone: str) -> str:
    return re.sub(r"\d", "", phone.upper())


def phone_class(phone: str) -> str:
    phone = clean_phone(phone)
    for name, group in [
        ("vowel", VOWELS),
        ("stop", STOPS),
        ("fricative", FRICATIVES),
        ("affricate", AFFRICATES),
        ("nasal", NASALS),
        ("liquid", LIQUIDS),
        ("glide", GLIDES),
    ]:
        if phone in group:
            return name
    return "other"


def normalise_word(token: str) -> str:
    match = WORD_RE.search(token)
    return match.group(0).lower() if match else ""


def flatten_words(result: dict, dictionary: dict) -> list[dict]:
    phones = []
    for segment in result.get("segments", []):
        for word in segment.get("words", []):
            token = normalise_word(word.get("word", ""))
            if not token or token not in dictionary:
                continue
            start, end = float(word["start"]), float(word["end"])
            pronunciation = dictionary[token][0]
            if end <= start or not pronunciation:
                continue
            step = (end - start) / len(pronunciation)
            for index, raw_phone in enumerate(pronunciation):
                phones.append({
                    "word": token,
                    "word_start_s": start,
                    "word_end_s": end,
                    "word_probability": float(word.get("probability", 0.0)),
                    "phone": clean_phone(raw_phone),
                    "phone_start_proxy_s": start + index * step,
                    "phone_end_proxy_s": start + (index + 1) * step,
                })
    return phones


def select_interval(phones: list[dict], duration: float) -> tuple[int, float, float]:
    candidates = []
    for index in range(1, len(phones) - 1):
        item = phones[index]
        centre = (item["phone_start_proxy_s"] + item["phone_end_proxy_s"]) / 2
        start, end = centre - 0.10, centre + 0.10
        if start < 0.25 or end > duration - 0.25:
            continue
        # Prefer a confidently recognised vowel near the middle of the clip.
        score = item["word_probability"] - 0.04 * abs(centre - duration / 2)
        if phone_class(item["phone"]) == "vowel":
            score += 0.5
        candidates.append((score, index, start, end))
    if not candidates:
        raise ValueError("No auditable central-phone proxy away from clip boundaries")
    _, index, start, end = max(candidates)
    return index, start, end


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/m3_pilot_manifest.csv")
    parser.add_argument("--aligned-manifest", default="data/m3_pilot_aligned_manifest.csv")
    parser.add_argument("--intervals", default="data/m3_pilot_intervals.csv")
    parser.add_argument("--asr-dir", default="outputs/m3_work/asr")
    parser.add_argument("--model", default="tiny.en")
    args = parser.parse_args()

    manifest_path = ROOT / args.manifest
    manifest = pd.read_csv(manifest_path, keep_default_na=False)
    asr_dir = ROOT / args.asr_dir
    asr_dir.mkdir(parents=True, exist_ok=True)
    dictionary = cmudict.dict()
    model = whisper.load_model(args.model, download_root=str(ROOT / "models" / "whisper"))
    intervals = []

    for row in manifest.itertuples(index=False):
        cache = asr_dir / f"{row.clip_id}.json"
        if cache.exists():
            result = json.loads(cache.read_text(encoding="utf-8"))
        else:
            print("transcribe", row.clip_id, flush=True)
            result = model.transcribe(
                str(ROOT / row.file_path), language="en", word_timestamps=True,
                fp16=False, verbose=False, temperature=0,
            )
            cache.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        phones = flatten_words(result, dictionary)
        duration = float(max((word["end"] for segment in result.get("segments", [])
                              for word in segment.get("words", [])), default=0.0))
        # The video may continue after speech. Use its feature timestamp as the
        # authoritative sampling duration so the selected context is in-bounds.
        features = pd.read_csv(ROOT / "outputs/m3_work/pilot_features/features" / f"{row.clip_id}.csv")
        duration = min(duration or float(features.time_s.max()), float(features.time_s.max()))
        index, start, end = select_interval(phones, duration)
        previous, central, following = phones[index - 1], phones[index], phones[index + 1]
        intervals.append({
            "clip_id": row.clip_id,
            "start_s": round(start, 6),
            "end_s": round(end, 6),
            "previous_phoneme": phone_class(previous["phone"]),
            "central_phoneme": phone_class(central["phone"]),
            "next_phoneme": phone_class(following["phone"]),
            "previous_phone_arpabet": previous["phone"],
            "central_phone_arpabet": central["phone"],
            "next_phone_arpabet": following["phone"],
            "central_word": central["word"],
            "word_probability": round(central["word_probability"], 6),
            "annotation_source": "whisper_tiny.en_word_timestamp+cmudict_uniform_phone_proxy",
            "alignment_quality": "weak_proxy_not_forced_alignment",
        })
        mask = manifest.clip_id.eq(row.clip_id)
        manifest.loc[mask, "transcript"] = str(result.get("text", "")).strip()
        manifest.loc[mask, "previous_phoneme"] = phone_class(previous["phone"])
        manifest.loc[mask, "central_phoneme"] = phone_class(central["phone"])
        manifest.loc[mask, "next_phoneme"] = phone_class(following["phone"])

    intervals_frame = pd.DataFrame(intervals)
    intervals_frame.to_csv(ROOT / args.intervals, index=False)
    manifest.to_csv(ROOT / args.aligned_manifest, index=False)
    print(intervals_frame[["previous_phoneme", "central_phoneme", "next_phoneme"]].apply(pd.Series.value_counts).fillna(0))
    print("aligned manifest", args.aligned_manifest)
    print("intervals", args.intervals)


if __name__ == "__main__":
    main()
