"""Expand the matched SadTalker course pilot with non-overlapping segments.

The same 12 licensed public speakers are retained.  Each speaker contributes
three 12-second segments (starts 5, 20 and 35 seconds); every real segment is
paired with a SadTalker clip generated from that segment's own audio and still
frame.  All segments from one speaker keep the original train/val/test split.

This is a course-scale variance-control experiment, not a claim of population
or generator generalisation.  The script is resumable and records provenance.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from build_matched_sadtalker_pilot import (
    PROCESSED_ROOT as BASE_PROCESSED_ROOT,
    RAW_ROOT as BASE_RAW_ROOT,
    ROOT,
    generate_fake,
    make_inputs,
    portable,
    sha256,
    standardise,
    transcode_real,
    validate_pair,
)


BASE_MANIFEST = ROOT / "data" / "matched_sadtalker_manifest.csv"
BASE_PROVENANCE = ROOT / "data" / "matched_sadtalker_provenance.json"
RAW_ROOT = ROOT / "data" / "raw" / "matched_sadtalker_expanded"
PROCESSED_ROOT = ROOT / "data" / "processed" / "matched_sadtalker_expanded"
DEFAULT_MANIFEST = ROOT / "data" / "expanded_sadtalker_manifest.csv"
DEFAULT_PROVENANCE = ROOT / "data" / "expanded_sadtalker_provenance.json"
STARTS = (5.0, 20.0, 35.0)


def source_path(source_stem: str) -> Path:
    matches = sorted((BASE_RAW_ROOT / "source").glob(f"{source_stem}.*"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one audited source for {source_stem}: {matches}")
    return matches[0]


def base_pair_rows(manifest: pd.DataFrame, speaker: str) -> list[dict]:
    selected = manifest.loc[manifest.speaker_id.eq(speaker)]
    if len(selected) != 2 or set(selected.label) != {"real", "fake"}:
        raise ValueError(f"Base matched pair is incomplete for {speaker}")
    return selected.to_dict("records")


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
        default=Path(os.environ.get("SADTALKER_WORK_ROOT", "/root/gpufree-data/expanded-sadtalker-work")),
    )
    args = parser.parse_args()

    base_manifest = pd.read_csv(BASE_MANIFEST, keep_default_na=False)
    base_provenance = json.loads(BASE_PROVENANCE.read_text(encoding="utf-8"))
    provenance_by_speaker = {item["speaker_id"]: item for item in base_provenance}
    speakers = (
        base_manifest[["speaker_id", "split"]].drop_duplicates()
        .sort_values(["split", "speaker_id"])
    )
    rows: list[dict] = []
    provenance: list[dict] = []

    for speaker_row in speakers.itertuples(index=False):
        speaker, split = speaker_row.speaker_id, speaker_row.split
        original = provenance_by_speaker[speaker]
        base_stem = original["pair_id"].removeprefix("matched_")
        raw_source = source_path(base_stem)

        # Segment zero is the already verified pair.  Reuse it byte-for-byte.
        for row in base_pair_rows(base_manifest, speaker):
            rows.append(dict(row))
        reused = dict(original)
        reused["segment_index"] = 0
        reused["start_s"] = STARTS[0]
        reused["expanded_status"] = "reused_verified_base_pair"
        provenance.append(reused)

        for segment_index, start_s in enumerate(STARTS[1:], start=1):
            pair_id = f"matched_{base_stem}_s{segment_index}"
            real_id, fake_id = f"matched_real_{base_stem}_s{segment_index}", f"matched_fake_{base_stem}_s{segment_index}"
            source_silent = RAW_ROOT / "source_silent" / f"{base_stem}_s{segment_index}.mp4"
            portrait = RAW_ROOT / "portrait" / f"{base_stem}_s{segment_index}.png"
            audio = RAW_ROOT / "audio" / f"{base_stem}_s{segment_index}.wav"
            generated_raw = RAW_ROOT / "generated" / f"{base_stem}_s{segment_index}.mp4"
            real_final = PROCESSED_ROOT / f"{real_id}.mp4"
            fake_final = PROCESSED_ROOT / f"{fake_id}.mp4"

            print(f"=== {pair_id} ===", flush=True)
            source_command = transcode_real(raw_source, source_silent, start_s, args.duration)
            make_inputs(source_silent, portrait, audio, args.duration)
            generation_command, generation_executed = generate_fake(
                portrait, audio, generated_raw, args.sadtalker_root,
                args.sadtalker_python, args.work_root,
            )
            real_command, real_executed = standardise(source_silent, audio, real_final, args.duration)
            fake_command, fake_executed = standardise(generated_raw, audio, fake_final, args.duration)
            real_probe, fake_probe, audio_pcm_md5 = validate_pair(
                real_final, fake_final, args.duration,
            )

            for clip_id, label, generator, path in [
                (real_id, "real", "none", real_final),
                (fake_id, "fake", "SadTalker_cd4c046", fake_final),
            ]:
                rows.append({
                    "clip_id": clip_id,
                    "file_path": path.relative_to(ROOT).as_posix(),
                    "label": label,
                    "speaker_id": speaker,
                    "generator": generator,
                    "transcript": "unknown",
                    "central_phoneme": "unknown",
                    "previous_phoneme": "unknown",
                    "next_phoneme": "unknown",
                    "compression": "paired_h264_crf23_512_25fps_aac16k",
                    "split": split,
                    "source_id": pair_id,
                })

            replacements = {
                raw_source: "<audited_public_source>",
                source_silent: "<matched_source_video>",
                portrait: "<source_portrait>",
                audio: "<shared_audio>",
                generated_raw: "<raw_sadtalker_output>",
                real_final: real_final.relative_to(ROOT).as_posix(),
                fake_final: fake_final.relative_to(ROOT).as_posix(),
            }
            provenance.append({
                "pair_id": pair_id,
                "speaker_id": speaker,
                "split": split,
                "segment_index": segment_index,
                "start_s": start_s,
                "duration_s": args.duration,
                "public_metadata": original["public_metadata"],
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
                    "sadtalker_generation": "executed" if generation_executed else "reused_verified_output",
                    "real_standardisation": "executed" if real_executed else "reused_verified_output",
                    "fake_standardisation": "executed" if fake_executed else "reused_verified_output",
                },
                "model": original["model"],
                "expanded_status": "new_non_overlapping_segment",
            })

            pd.DataFrame(rows).sort_values(["split", "speaker_id", "source_id", "label"]).to_csv(
                args.manifest, index=False,
            )
            args.provenance.write_text(json.dumps(provenance, indent=2), encoding="utf-8")

    frame = pd.DataFrame(rows).sort_values(["split", "speaker_id", "source_id", "label"])
    if len(frame) != 72 or frame.source_id.nunique() != 36:
        raise AssertionError("Expanded pilot must contain 36 complete pairs / 72 clips")
    if (frame.groupby("speaker_id").split.nunique() != 1).any():
        raise AssertionError("A speaker must never cross splits")
    frame.to_csv(args.manifest, index=False)
    args.provenance.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(frame.groupby(["split", "label"]).size())
    print("manifest", args.manifest)
    print("provenance", args.provenance)


if __name__ == "__main__":
    main()
