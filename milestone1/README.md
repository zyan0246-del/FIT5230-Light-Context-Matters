# FIT5230 Milestone 1 implementation

This folder contains the executable Milestone 1 materials for the Theme 3 Light-side project.

## Files

- `milestone1.ipynb` - the team's initial customized notebook for Google Colab.
- `demo_outputs/` - verified smoke-test figures, CSVs, and JSON produced from public media.
- `Public_Demo_Sources.md` - source, attribution, and interpretation limits for the public demo.
- `Interactive_Challenge.md` - detailed challenge rules for current and later milestones.

## Reproduce the smoke test

1. Upload `milestone1.ipynb` to Google Colab.
2. Leave `USE_PUBLIC_DEMO = True` and run all cells once. The notebook automatically downloads a licensed Wikitongues real clip, a public TalkingHeadBench generated clip, and the MediaPipe model (about 20 MB total before the short real clip is transcoded).
3. Confirm that the notebook produces the quality-control table, landmark previews, trajectory plots, CSV features, summary JSON, and environment metadata.
4. Run `python milestone1/verify_milestone1.py` from the repository root to check the committed technical artifacts.

## Colab verification checklist

Before treating a run as complete, confirm that:

- the installation and import cells complete successfully;
- the input quality-control table shows the expected resolution, duration, and frame rate;
- landmark coverage is preferably at least 0.80 for both clips;
- `milestone1_outputs/landmark_preview_real.png` clearly overlays the mouth;
- `milestone1_outputs/lip_dynamics_comparison.png` contains labeled real/generated traces;
- the export cell creates CSV, JSON, and environment files;
- all required cells retain their visible outputs when the notebook is saved.

The public pair has already passed the end-to-end pipeline with 180 sampled real frames and 149 sampled generated frames. The committed `demo_outputs/landmark_previews.png` and `demo_outputs/lip_dynamics_comparison.png` provide visible evidence that the customized pipeline runs.

The two public clips differ in identity, speech, resolution, and capture conditions. Treat their outputs only as evidence that the software works. Do **not** describe their numerical differences as evidence that the proposed detector works. A controlled same-audio design is still required for later research claims.

Do not state that PIA has been reproduced unless the team has actually run it. The current milestone uses PIA as the main technical baseline, CALS as theoretical motivation, and GenVidBench as a general video-detection reference.

## Scope control

Milestone 1 presents coarticulation as a testable research hypothesis. The minimum course-project pipeline remains viable even if that cue is weak: the verified lip-dynamics features can support a self-contained temporal classifier, with PIA and general visual evidence added as comparisons or fusion branches after they are reproduced. This avoids making the course deliverable depend on an unverified signal or a single external repository.
