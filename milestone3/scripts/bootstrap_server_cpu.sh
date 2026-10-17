#!/usr/bin/env bash
set -euo pipefail

# Prepare the public M3 pipeline on a CPU-only Ubuntu server.  This script does
# not install CUDA, PyTorch or a talking-face generator; those are deliberately
# deferred until a GPU is attached so that the base environment stays portable.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
M3_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_PATH="${M3_VENV_PATH:-${M3_ROOT}/.venv}"
DATA_ROOT="${M3_DATA_ROOT:-${M3_ROOT}/server_data}"
LOG_ROOT="${M3_LOG_ROOT:-${M3_ROOT}/server_logs}"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "This bootstrap targets an Ubuntu/Debian Linux server." >&2
  exit 1
fi

if [[ "${M3_SKIP_APT:-0}" == "1" ]]; then
  echo "Skipping apt installation because M3_SKIP_APT=1."
elif command -v apt-get >/dev/null 2>&1; then
  if [[ "${EUID}" -eq 0 ]]; then
    SUDO=()
  elif command -v sudo >/dev/null 2>&1; then
    SUDO=(sudo)
  else
    echo "apt-get needs root access, but sudo is unavailable." >&2
    exit 1
  fi
  "${SUDO[@]}" apt-get update
  DEBIAN_FRONTEND=noninteractive "${SUDO[@]}" apt-get install -y \
    build-essential ffmpeg git libgl1 libglib2.0-0 python3 python3-dev \
    python3-pip python3-venv
else
  echo "apt-get was not found. Install Python 3.10-3.13, venv, FFmpeg and Git first." >&2
  exit 1
fi

mkdir -p \
  "${DATA_ROOT}/raw_real" \
  "${DATA_ROOT}/generated_fake" \
  "${DATA_ROOT}/processed" \
  "${DATA_ROOT}/manifests" \
  "${DATA_ROOT}/cache" \
  "${LOG_ROOT}"

VENV_ARGS=()
if [[ "${M3_USE_SYSTEM_SITE_PACKAGES:-0}" == "1" ]]; then
  VENV_ARGS+=(--system-site-packages)
fi
"${M3_PYTHON_BIN:-python3}" -m venv "${VENV_ARGS[@]}" "${VENV_PATH}"
# shellcheck disable=SC1091
source "${VENV_PATH}/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r "${M3_ROOT}/requirements.txt"
python -m pip install -e "${M3_ROOT}" --no-deps

cd "${M3_ROOT}"
python -m s2f.cli smoke
python -m pytest -q

{
  echo "utc_time=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "hostname=$(hostname)"
  echo "python=$(python --version 2>&1)"
  echo "ffmpeg=$(ffmpeg -version | head -n 1)"
  echo "venv=${VENV_PATH}"
  echo "data_root=${DATA_ROOT}"
  echo "git_commit=$(git rev-parse HEAD 2>/dev/null || echo unavailable)"
} | tee "${LOG_ROOT}/cpu_environment.txt"

if [[ "${M3_DOWNLOAD_PUBLIC:-0}" == "1" ]]; then
  python scripts/download_m3_pilot.py
fi

echo
echo "CPU environment is ready."
echo "Activate it with: source ${VENV_PATH}/bin/activate"
echo "Data directory: ${DATA_ROOT}"
echo "Environment record: ${LOG_ROOT}/cpu_environment.txt"
