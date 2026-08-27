#!/usr/bin/env python3
"""Offline verification for the public FIT5230 Milestone 1 artifacts."""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"


results: list[tuple[str, str, str]] = []


def record(status: str, check: str, detail: str) -> None:
    results.append((status, check, detail))


def require_file(relative_path: str) -> Path:
    path = ROOT / relative_path
    if path.is_file() and path.stat().st_size > 0:
        record(PASS, f"file:{relative_path}", f"{path.stat().st_size} bytes")
    else:
        record(FAIL, f"file:{relative_path}", "missing or empty")
    return path


required_files = [
    "milestone1.ipynb",
    "Interactive_Challenge.md",
    "Public_Demo_Sources.md",
    "demo_outputs/landmark_previews.png",
    "demo_outputs/lip_dynamics_comparison.png",
    "demo_outputs/summary.json",
    "demo_outputs/environment.json",
]

for required_file in required_files:
    require_file(required_file)


notebook_path = ROOT / "milestone1.ipynb"
if notebook_path.is_file():
    try:
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        record(PASS, "notebook:JSON", f"{len(notebook.get('cells', []))} cells")
        parsed_cells = 0
        for index, cell in enumerate(notebook.get("cells", [])):
            if cell.get("cell_type") != "code":
                continue
            source_lines = [
                line
                for line in cell.get("source", [])
                if not line.lstrip().startswith(("%", "!"))
            ]
            source = "".join(source_lines).strip()
            if source:
                ast.parse(source, filename=f"cell_{index}")
                parsed_cells += 1
        record(PASS, "notebook:Python syntax", f"{parsed_cells} code cells parsed")
    except (json.JSONDecodeError, SyntaxError) as error:
        record(FAIL, "notebook:structure/syntax", str(error))


summary_path = ROOT / "demo_outputs/summary.json"
if summary_path.is_file():
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        expected_labels = {"real", "generated"}
        if not expected_labels.issubset(summary):
            record(FAIL, "demo:labels", f"found {sorted(summary)}")
        else:
            record(PASS, "demo:labels", "real and generated")
            for label in sorted(expected_labels):
                coverage = float(summary[label]["landmark_coverage"])
                frames = int(summary[label]["sampled_frames"])
                status = PASS if coverage >= 0.80 and frames > 0 else FAIL
                record(
                    status,
                    f"demo:{label}",
                    f"{frames} sampled frames, landmark coverage={coverage:.1%}",
                )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        record(FAIL, "demo:summary", str(error))


for status, check, detail in results:
    print(f"[{status}] {check}: {detail}")

counts = {status: sum(item[0] == status for item in results) for status in (PASS, WARN, FAIL)}
print(
    f"\nSummary: {counts[PASS]} passed, {counts[WARN]} warnings, "
    f"{counts[FAIL]} failed."
)
print("The verifier checks the committed technical artifacts.")

sys.exit(1 if counts[FAIL] else 0)
