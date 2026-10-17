#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
M3_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_PATH="${M3_VENV_PATH:-${M3_ROOT}/.venv}"
LOG_ROOT="${M3_LOG_ROOT:-${M3_ROOT}/server_logs}"

mkdir -p "${LOG_ROOT}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "No NVIDIA driver is visible. Attach a GPU and install the provider's driver image first." >&2
  exit 2
fi

nvidia-smi | tee "${LOG_ROOT}/nvidia-smi.txt"

if [[ -x "${VENV_PATH}/bin/python" ]]; then
  "${VENV_PATH}/bin/python" - <<'PY' | tee "${LOG_ROOT}/python_gpu_check.txt"
import importlib.util
import platform

print("python:", platform.python_version())
if importlib.util.find_spec("torch") is None:
    print("torch: not installed (expected before the generator environment is added)")
else:
    import torch
    print("torch:", torch.__version__)
    print("cuda_available:", torch.cuda.is_available())
    print("cuda_runtime:", torch.version.cuda)
    print("device_count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("device_0:", torch.cuda.get_device_name(0))
PY
else
  echo "M3 virtual environment not found at ${VENV_PATH}; run bootstrap_server_cpu.sh first." >&2
  exit 3
fi

echo "GPU visibility check completed. Install the generator in a separate environment after choosing the GPU image."
