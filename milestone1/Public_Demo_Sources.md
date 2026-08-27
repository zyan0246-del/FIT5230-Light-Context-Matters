# Public demo sources and interpretation limits

The notebook uses these small public files when `USE_PUBLIC_DEMO = True`.

## Real talking-face video

- Title: *WIKITONGUES: Simon speaking Cumbrian*
- Creator/attribution: Wikitongues
- Source: https://commons.wikimedia.org/wiki/File:WIKITONGUES-_Simon_speaking_Cumbrian.webm
- License stated on the source page: Creative Commons Attribution 3.0 (CC BY 3.0)
- Notebook use: downloads the source WebM and transcodes seconds 2-14 to MP4.

## Generated talking-face video

- Dataset: TalkingHeadBench
- Generator subset: Hallo
- Sample: `00023--7lHr2ZNlULA_4--Hallo.mp4`
- Dataset card: https://huggingface.co/datasets/luchaoqi/TalkingHeadBench
- Direct file used by the notebook: https://huggingface.co/datasets/luchaoqi/TalkingHeadBench/resolve/main/fake/Hallo/test/00023--7lHr2ZNlULA_4--Hallo.mp4?download=true

## What the demo proves

The verified run shows that the notebook can download/transcode video, open real and generated MP4 inputs, track mouth landmarks, compute normalized temporal features, plot trajectories, and export reproducible files.

## What the demo does not prove

The clips differ in identity, speech, resolution, pose, lighting, and capture conditions. Their curves are therefore confounded and must not be used to claim detection accuracy or a genuine-versus-generated causal difference. Later experiments need controlled same-audio or phoneme-aligned pairs, multiple identities and generators, and speaker-/generator-disjoint evaluation.
