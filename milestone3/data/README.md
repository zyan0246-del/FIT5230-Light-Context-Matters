# Data contracts

Manifest CSV columns are also described by `manifest.schema.json`. Paths are relative to project root. One row represents one clip, never a frame. Files are checked on disk; duplicate clip IDs, normalized paths and SHA256 contents are rejected. Source-family leakage is always rejected. Speaker/generator checks depend on the explicitly selected protocol.

| Column | Contract |
|---|---|
| clip_id | Stable letters/digits/underscore/hyphen; unique |
| file_path | Project-relative MP4 path; synthetic smoke explicitly uses CSV fixtures |
| label | real or fake |
| speaker_id | Stable identity ID, shared by genuine/generated versions |
| generator | Generator family for fake; `none` for real |
| transcript | Actual speech/prompt or `unknown` |
| central_phoneme, previous_phoneme, next_phoneme | Consistent phone strings; `unknown` until annotated |
| compression | Codec/CRF/FPS/resolution condition |
| split | train, val or test; chosen before modeling |
| source_id | Required common source-family ID for original, fake, edits and derivatives |

Use one source_id for all versions derived from the same source recording/audio. Speaker IDs and source IDs cannot be inferred reliably from filenames: curate them. SHA256 catches byte duplicates but cannot detect every re-encoded duplicate; provenance grouping is essential. Do not treat separate clips of one recording as independent evidence.

Phoneme intervals use `clip_id,start_s,end_s,previous_phoneme,central_phoneme,next_phoneme,annotation_source`. Times are relative to the analyzed MP4 including any trimming. Intervals are half-open `[start_s,end_s)` and use a default 0.2-second before/after window. At least five sampled frames per part and adequate coverage are required. One annotated interval per clip is supported in v1. This also supports words by using word strings consistently, but such runs must be described as word-context experiments.

For real evaluation, annotate without consulting labels wherever possible, and match central phoneme/context distributions across classes. Otherwise the classifier can learn transcription/source shortcuts. Do not invent annotations for public demo clips. `Aligner.align(video_path, transcript)` in `src/s2f/features.py` is a future automatic-alignment interface; no automatic implementation is installed.

The source file for metadata is the original MP4. Per-frame tables are under `outputs/<run>/features/`; summary feature tables and metrics are in separate folders. Never pass M1 CSVs directly into M2 evaluation: pixel-aspect correction and gap handling change feature semantics. Regenerate them from the source videos.
