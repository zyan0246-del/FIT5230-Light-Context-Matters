#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
M3_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${M3_PYTHON_BIN:-/root/gpufree-data/m3-venv/bin/python}"

cd "${M3_ROOT}"

manifest="data/expanded_sadtalker_manifest.csv"
aligned_manifest="data/expanded_sadtalker_aligned_manifest.csv"
intervals="data/expanded_sadtalker_intervals.csv"
feature_root="outputs/expanded_sadtalker"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "M3 Python environment not found: ${PYTHON_BIN}" >&2
  exit 2
fi
if [[ ! -f "${manifest}" ]]; then
  echo "Expanded manifest is missing; run build_expanded_sadtalker_pilot.py first." >&2
  exit 3
fi

"${PYTHON_BIN}" -m s2f.cli validate-manifest "${manifest}" --protocol speaker
if "${PYTHON_BIN}" - "${feature_root}" <<'PY'
import sys
from pathlib import Path
import pandas as pd

root = Path(sys.argv[1])
features = list((root / "features").glob("*.csv"))
qc_path = root / "qc.csv"
if len(features) != 72 or not qc_path.exists():
    raise SystemExit(1)
qc = pd.read_csv(qc_path)
if len(qc) != 72 or (qc["status"] != "ok").any() or float(qc["coverage"].min()) < 0.8:
    raise SystemExit(1)
print("Reusing 72 audited landmark files; minimum coverage", float(qc["coverage"].min()))
PY
then
  :
else
  "${PYTHON_BIN}" -m s2f.cli extract-manifest "${manifest}" --out "${feature_root}"
fi

"${PYTHON_BIN}" scripts/build_weak_phoneme_intervals.py \
  --manifest "${manifest}" \
  --aligned-manifest "${aligned_manifest}" \
  --intervals "${intervals}" \
  --asr-dir "outputs/expanded_sadtalker/asr" \
  --feature-dir "outputs/expanded_sadtalker/features" \
  --model tiny.en \
  --reuse-source-asr

"${PYTHON_BIN}" -m s2f.cli validate-manifest "${aligned_manifest}" --protocol speaker
"${PYTHON_BIN}" -m s2f.cli evaluate "${aligned_manifest}" \
  --features "outputs/expanded_sadtalker/features" \
  --intervals "${intervals}" \
  --out "outputs/expanded_sadtalker/evaluation" \
  --protocol speaker

PYTHONPATH=src "${PYTHON_BIN}" scripts/run_viseme_context_optimization.py \
  --manifest "${aligned_manifest}" \
  --intervals "${intervals}" \
  --features "outputs/expanded_sadtalker/features" \
  --out "artifacts/viseme_optimization_expanded"

PYTHONPATH=src "${PYTHON_BIN}" scripts/run_viseme_context_optimization.py \
  --manifest "${aligned_manifest}" \
  --intervals "${intervals}" \
  --features "outputs/expanded_sadtalker/features" \
  --out "artifacts/viseme_optimization_new_segments" \
  --new-segments-only

"${PYTHON_BIN}" -m pytest -q
echo "Expanded matched SadTalker course pipeline completed."
