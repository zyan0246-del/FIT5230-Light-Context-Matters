"""Build the standalone, inspectable FIT5230 Milestone 3 notebook."""
from __future__ import annotations

from pathlib import Path
import hashlib
import json

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]


def text_snapshot() -> tuple[dict[str, str], str]:
    paths = list((ROOT / "src/s2f").glob("*.py"))
    paths += list((ROOT / "tests").glob("*.py"))
    paths += list((ROOT / "challenge").glob("*.csv"))
    paths += [
        ROOT / "requirements.txt",
        ROOT / "configs/default.json",
        ROOT / "data/m3_pilot_sources.json",
        ROOT / "data/m3_pilot_manifest.csv",
        ROOT / "data/m3_pilot_provenance.json",
        ROOT / "data/manifest.schema.json",
        ROOT / "scripts/download_m3_pilot.py",
        ROOT / "scripts/audit_m3_pilot.py",
        ROOT / "scripts/build_weak_phoneme_intervals.py",
        ROOT / "scripts/summarize_lip_pilot.py",
        ROOT / "scripts/run_chromatic_robustness.py",
    ]
    paths += list((ROOT / "artifacts/m3_results").glob("*.csv"))
    paths += list((ROOT / "artifacts/m3_results").glob("*.json"))
    sources = {
        path.relative_to(ROOT).as_posix(): path.read_text(encoding="utf-8")
        for path in sorted(set(paths))
    }
    serialised = json.dumps(sources, sort_keys=True)
    return sources, hashlib.sha256(serialised.encode()).hexdigest()


def build() -> None:
    sources, digest = text_snapshot()
    bootstrap = f'''from pathlib import Path
import hashlib, json, os, subprocess, sys
SOURCES = {sources!r}
EXPECTED_SHA256 = {digest!r}
assert hashlib.sha256(json.dumps(SOURCES, sort_keys=True).encode()).hexdigest() == EXPECTED_SHA256
ROOT = Path.cwd() / "m3_notebook_workspace"
ROOT.mkdir(exist_ok=True)
for relative, content in SOURCES.items():
    target = ROOT / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.read_text(encoding="utf-8") != content:
        raise RuntimeError(f"Refusing to overwrite a changed file: {{target}}")
    target.write_text(content, encoding="utf-8")
os.environ["MPLCONFIGDIR"] = str(ROOT / "outputs/.mplcache")
os.environ["PYTHONPATH"] = str(ROOT / "src")
sys.path.insert(0, str(ROOT / "src"))
print("Embedded source SHA256:", EXPECTED_SHA256)
print("Inspectable text files:", len(SOURCES))
'''
    install = '''import importlib.metadata as metadata
from packaging.requirements import Requirement
requirements = [Requirement(line) for line in (ROOT / "requirements.txt").read_text().splitlines()
                if line.strip() and not line.lstrip().startswith("#")]
missing = []
for requirement in requirements:
    try:
        if metadata.version(requirement.name) not in requirement.specifier:
            missing.append(str(requirement))
    except metadata.PackageNotFoundError:
        missing.append(str(requirement))
if missing:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(ROOT / "requirements.txt")], check=True)
else:
    print("Declared package versions already satisfied.")
from s2f.demo import environment
print(json.dumps(environment(), indent=2))
'''
    cells = [
        nbf.v4.new_markdown_cell(
            "# Context Matters: FIT5230 Milestone 3\n\n"
            "**Light.Ningyizhuo_No.1 | Yan Zhiheng and He Hongjie | Theme 3, Light side**\n\n"
            "This notebook evaluates two distinct defenses. Yan tests whether weak phonetic context improves "
            "mouth-dynamics detection. He audits the periodic colour trace introduced by Dark.Mimic's rPPG evasion. "
            "The public pilot runs end to end on CPU. Peer videos remain private, so the notebook includes their hashes, "
            "aggregate results and a guarded rerun path without redistributing the MP4 files."
        ),
        nbf.v4.new_markdown_cell(
            "## 1. Reproducible source and environment\n\n"
            "The notebook contains a readable source snapshot and verifies its SHA256 before execution. "
            "A fresh run downloads approximately 180 MB of public models and videos. No paid GPU is required."
        ),
        nbf.v4.new_code_cell(bootstrap),
        nbf.v4.new_code_cell(install),
        nbf.v4.new_markdown_cell(
            "## 2. Regression tests\n\n"
            "Tests cover manifest leakage, missing landmark runs, context representation, challenge input, "
            "paired score handling and chromatic spectral calculations."
        ),
        nbf.v4.new_code_cell(
            '''test_temp = ROOT / "tmp/pytest"
test_temp.parent.mkdir(parents=True, exist_ok=True)
result = subprocess.run(
    [sys.executable, "-m", "pytest", "-q", "--tb=short", "--basetemp", str(test_temp), "tests"],
    cwd=ROOT, env=dict(os.environ, PYTHONPATH=str(ROOT / "src")), text=True, capture_output=True,
)
print(result.stdout)
print(result.stderr)
result.check_returncode()
'''
        ),
        nbf.v4.new_markdown_cell(
            "## 3. Public 24-clip pilot\n\n"
            "The pilot contains 12 licensed Wikitongues real clips and 12 TalkingHeadBench fakes "
            "from Hallo and LivePortrait. Splits are fixed before modelling and remain speaker/source disjoint. "
            "Real and fake clips come from different collections, so this experiment can establish feasibility only."
        ),
        nbf.v4.new_code_cell(
            '''processed = ROOT / "data/processed/m3_pilot"
if len(list(processed.glob("*.mp4"))) != 24:
    subprocess.run([sys.executable, "scripts/download_m3_pilot.py"], cwd=ROOT, check=True)
else:
    print("Reusing 24 hash-recorded public pilot clips from the execution cache.")
subprocess.run([sys.executable, "scripts/audit_m3_pilot.py"], cwd=ROOT, check=True)
import pandas as pd
from IPython.display import display, Image
manifest = pd.read_csv(ROOT / "data/m3_pilot_manifest.csv")
display(manifest.groupby(["split", "label"]).size().rename("clips").reset_index())
display(pd.read_csv(ROOT / "outputs/m3_work/pilot_qc/metadata_qc.csv").head())
display(Image(filename=str(ROOT / "outputs/m3_work/pilot_qc/contact_sheet.png")))
'''
        ),
        nbf.v4.new_markdown_cell(
            "## 4. Mouth landmarks and phonetic context\n\n"
            "MediaPipe extracts normalized lip geometry and derivatives. Whisper tiny.en supplies word timestamps; "
            "CMUdict supplies English pronunciations. Uniform phone placement within each word creates a weak proxy, "
            "not forced-alignment ground truth. We compare both models on the same 12 train, 6 validation and 6 test clips."
        ),
        nbf.v4.new_code_cell(
            '''from s2f.demo import MODEL_URL, download
def run_checked(command):
    completed = subprocess.run(command, cwd=ROOT, check=False, text=True, capture_output=True,
                               env=dict(os.environ, PYTHONPATH=str(ROOT / "src")))
    if completed.returncode:
        print(completed.stdout)
        print(completed.stderr)
        completed.check_returncode()
    if completed.stdout.strip():
        print(completed.stdout.strip())

model = download(MODEL_URL, ROOT / "models/face_landmarker.task")
features = ROOT / "outputs/m3_work/pilot_features"
if not (features / "qc.csv").exists():
    run_checked([sys.executable, "-m", "s2f.cli", "extract-manifest", "data/m3_pilot_manifest.csv",
                 "--root", ".", "--out", "outputs/m3_work/pilot_features",
                 "--model", "models/face_landmarker.task"])
qc = pd.read_csv(features / "qc.csv")
display(qc[["clip_id", "sampled_frames", "coverage", "status"]])
assert len(qc) == 24 and qc.coverage.min() >= 0.8
run_checked([sys.executable, "scripts/build_weak_phoneme_intervals.py"])
run_checked([sys.executable, "-m", "s2f.cli", "evaluate", "data/m3_pilot_aligned_manifest.csv",
             "--features", "outputs/m3_work/pilot_features/features", "--intervals", "data/m3_pilot_intervals.csv",
             "--root", ".", "--out", "outputs/m3_work/pilot_eval_context", "--protocol", "speaker"])
run_checked([sys.executable, "scripts/summarize_lip_pilot.py"])
'''
        ),
        nbf.v4.new_code_cell(
            '''results = pd.read_csv(ROOT / "outputs/m3_work/pilot_eval_context/metrics/results.csv")
display(results)
display(pd.read_csv(ROOT / "outputs/m3_work/lip_pilot_summary/held_out_predictions_and_failures.csv"))
display(Image(filename=str(ROOT / "outputs/m3_work/lip_pilot_summary/lip_pilot_results.png")))
print("Result: the weak context proxy did not improve the six-clip held-out pilot.")
'''
        ),
        nbf.v4.new_markdown_cell(
            "## 5. Dark.Mimic paired chromatic audit\n\n"
            "Dark.Mimic supplied one aligned control and A=1 modified video. We measure modified-minus-control RGB "
            "differences and spectral energy near the peer-reported 1.2 Hz setting. A neutral re-encode tests codec effects. "
            "The raw peer files stay outside the public repository. The aggregate below identifies them by SHA256. "
            "This paired forensic measurement does not estimate physiology and does not classify real versus fake."
        ),
        nbf.v4.new_code_cell(
            '''provenance = json.loads((ROOT / "artifacts/m3_results/peer_pair_provenance.json").read_text())
print(json.dumps(provenance, indent=2))
pair = pd.read_csv(ROOT / "artifacts/m3_results/attack_vs_neutral.csv")
robust = pd.read_csv(ROOT / "artifacts/m3_results/robustness_summary.csv")
display(pair[pair.roi.eq("global")])
display(robust)

import matplotlib.pyplot as plt
red = robust[robust.channel.eq("red")].set_index("condition").loc[["original", "h264_crf35", "resize_256", "fps_15"]]
x = range(len(red))
fig, ax = plt.subplots(figsize=(9, 4.6))
ax.bar([i - .18 for i in x], red.attack, .36, label="Dark.Mimic A=1")
ax.bar([i + .18 for i in x], red.neutral, .36, label="Neutral re-encode")
ax.set_xticks(list(x), ["Original", "H.264 CRF 35", "256×224", "15 FPS"])
ax.set_ylim(0, 1); ax.set_ylabel("Red energy fraction near 1.2 Hz")
ax.set_title("Periodic chromatic trace under post-processing"); ax.legend(); ax.grid(axis="y", alpha=.2)
plt.show()
'''
        ),
        nbf.v4.new_markdown_cell(
            "### Optional private-pair rerun\n\n"
            "Team members can place the three verified MP4 files under `data/private/dark_mimic/` and set "
            "`RUN_PRIVATE_PAIR=True`. The public notebook skips this cell rather than downloading or fabricating private data."
        ),
        nbf.v4.new_code_cell(
            '''RUN_PRIVATE_PAIR = False
private = ROOT / "data/private/dark_mimic"
required_private = [private / "person01_fake01_control.mp4",
                    private / "person01_fake01_attacked_A1.mp4",
                    private / "person01_fake01_control_neutral_reencode.mp4"]
if RUN_PRIVATE_PAIR:
    if not all(path.exists() for path in required_private):
        raise FileNotFoundError("Place the three hash-verified private MP4 files in data/private/dark_mimic")
    subprocess.run([sys.executable, "scripts/run_chromatic_robustness.py"], cwd=ROOT, check=True,
                   env=dict(os.environ, PYTHONPATH=str(ROOT / "src")))
else:
    print("Private-pair rerun skipped. Displayed aggregate results remain linked to SHA256 provenance.")
'''
        ),
        nbf.v4.new_markdown_cell(
            "## 6. Conclusions and claim boundaries\n\n"
            "1. All 24 public clips passed landmark coverage, but the context-free test result remained small and uncertain "
            "(macro-F1 0.486; AUROC 0.667; n=6).\n"
            "2. The weak phonetic-context proxy did not improve the held-out result (macro-F1 0.486; AUROC 0.556). "
            "This negative result points to forced alignment, matched real/fake sources and more clips as the next research step.\n"
            "3. In the single peer pair, periodic colour energy near 1.2 Hz remained higher than the neutral control after "
            "H.264 CRF 35, resizing and 15 FPS. This supports a paired manipulation audit, not a general detector.\n\n"
            "The integrated Light defense keeps both channels separate: lip dynamics test S2F generation artefacts, while "
            "chromatic forensics test traces introduced by the attempted rPPG evasion. We do not tune fusion on one peer pair."
        ),
    ]
    notebook = nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
            "source_snapshot_sha256": digest,
        },
    )
    destination = ROOT / "notebooks/Milestone3.ipynb"
    destination.parent.mkdir(exist_ok=True)
    nbf.write(notebook, destination)
    print("Built", destination, "with", len(cells), "cells and source", digest)


if __name__ == "__main__":
    build()
