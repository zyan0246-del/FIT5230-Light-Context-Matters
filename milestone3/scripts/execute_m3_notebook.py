"""Execute Milestone3.ipynb in an isolated local workspace and save outputs."""
from pathlib import Path
import json
import os
import shutil
import sys
import tempfile

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager


ROOT = Path(__file__).resolve().parents[1]
KERNELS = ROOT / "tmp/kernels"
SPEC = KERNELS / "s2f-m3"
SPEC.mkdir(parents=True, exist_ok=True)
(SPEC / "kernel.json").write_text(json.dumps({
    "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
    "display_name": "S2F M3 local verification", "language": "python",
}), encoding="utf-8")
os.environ["JUPYTER_RUNTIME_DIR"] = str(ROOT / "tmp/jupyter-m3")
Path(os.environ["JUPYTER_RUNTIME_DIR"]).mkdir(parents=True, exist_ok=True)
manager = KernelManager(
    kernel_name="s2f-m3",
    kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(KERNELS)]),
)
notebook_path = ROOT / "notebooks/Milestone3.ipynb"
notebook = nbformat.read(notebook_path, as_version=4)
run_dir = Path(tempfile.mkdtemp(prefix="m3_notebook_run_", dir=ROOT / "tmp"))
workspace = run_dir / "m3_notebook_workspace"

# Reuse independently acquired caches. A fresh Colab run uses the same public
# acquisition scripts and URLs instead.
for relative in ["data/raw/m3_pilot", "data/processed/m3_pilot", "models"]:
    source = ROOT / relative
    if source.exists():
        shutil.copytree(source, workspace / relative)

client = NotebookClient(
    notebook, km=manager, timeout=1200,
    resources={"metadata": {"path": str(run_dir)}}, allow_errors=False,
)
try:
    client.execute()
finally:
    nbformat.write(notebook, notebook_path)
    if manager.has_kernel:
        manager.shutdown_kernel(now=True)
code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
if any(not cell.get("outputs") for cell in code_cells):
    raise RuntimeError("Every code cell must retain an execution output")
print("Executed notebook with saved outputs:", len(code_cells), "code cells")
