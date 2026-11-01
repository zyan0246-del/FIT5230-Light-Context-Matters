# FIT5230 M4 Strategy A public demo

This directory contains Yan Zhiheng's individual Milestone 4 proof-of-concept code and public feature tables.

## Run in Google Colab

Open `M4_Strategy_A_Demo.ipynb` and select **Runtime -> Run all**. The notebook clones this repository, installs four Python packages, runs both demonstrations, and displays their summaries and figures.

## Run locally

```bash
python -m pip install -r code/requirements.txt
python code/run_m4_pocs.py
python code/verify_m4_demo.py
```

The script reads only `demo_data/`. No private peer video is required or distributed.

The verifier checks the input SHA256 values, fixed seed/fold contract, reported
claim boundaries, repeat-level attack intervals, and monotonic attack-budget
response. A successful run prints `M4 public demo verification: all checks passed.`

## Evidence boundary

- The enhancement demo uses 24 public clips and 547 speech events.
- Weighting alone is retained as a negative control; the robust follow-up improves both means but is post-hoc.
- The role-reversal sweep is feature-space; a separate saved summary records one rendered and re-extracted MP4 case.
- The rendered source video is not redistributed; this prevents the public notebook from rerendering that case by default.
- Bootstrap intervals resample the 20 repeat seeds, rather than treating repeated clip predictions as independent subjects.
