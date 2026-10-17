#!/usr/bin/env bash
set -euo pipefail

SADTALKER_ROOT="${SADTALKER_ROOT:-/root/gpufree-data/generators/SadTalker-cd4c046}"
SADTALKER_PYTHON="${SADTALKER_PYTHON:-/root/gpufree-data/sadtalker-env/bin/python}"
RESULT_ROOT="${SADTALKER_RESULT_ROOT:-/root/gpufree-data/sadtalker-smoke-results}"
LOG_ROOT="${SADTALKER_LOG_ROOT:-/root/gpufree-data/m3-logs}"

if [[ "${M3_GPU_MODE_CONFIRMED:-0}" != "1" ]]; then
  echo "Refusing to generate until paid/dedicated GPU mode is confirmed." >&2
  echo "After starting the GPU instance, rerun with M3_GPU_MODE_CONFIRMED=1." >&2
  exit 2
fi

for command_name in nvidia-smi ffmpeg ffprobe sha256sum; do
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    echo "Required command not found: ${command_name}" >&2
    exit 3
  fi
done

if [[ ! -x "${SADTALKER_PYTHON}" ]]; then
  echo "SadTalker Python environment not found: ${SADTALKER_PYTHON}" >&2
  exit 4
fi

if [[ ! -f "${SADTALKER_ROOT}/inference.py" ]]; then
  echo "SadTalker checkout not found: ${SADTALKER_ROOT}" >&2
  exit 5
fi

mkdir -p "${RESULT_ROOT}" "${LOG_ROOT}"

free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
if [[ ! "${free_mib}" =~ ^[0-9]+$ ]] || (( free_mib < 6000 )); then
  echo "At least 6000 MiB of free GPU memory is required; detected ${free_mib:-unknown} MiB." >&2
  exit 6
fi

nvidia-smi | tee "${LOG_ROOT}/sadtalker_nvidia-smi.txt"

"${SADTALKER_PYTHON}" - <<'PY' | tee "${LOG_ROOT}/sadtalker_gpu_check.txt"
import torch

print("torch:", torch.__version__)
print("cuda_runtime:", torch.version.cuda)
print("cuda_available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available to the SadTalker environment")
print("device_0:", torch.cuda.get_device_name(0))
PY

cd "${SADTALKER_ROOT}"
sha256sum -c <<'EOF'
c211f5d6de003516bf1bbda9f47049a4c9c99133b1ab565c6961e5af16477bff  checkpoints/SadTalker_V0.0.2_256.safetensors
62a1e06006cc963220f6477438518ed86e9788226c62ae382ddc42fbcefb83f1  checkpoints/mapping_00229-model.pth.tar
84a8642468a3fcfdd9ab6be955267043116c2bec2284686a5262f1eaf017f64c  checkpoints/mapping_00109-model.pth.tar
bbfd137307a4c7debd5c283b9b0ce539466cee417ac0a155e184d857f9f2899c  gfpgan/weights/alignment_WFLW_4HG.pth
6d1de9c2944f2ccddca5f5e010ea5ae64a39845a86311af6fdf30841b0a5a16d  gfpgan/weights/detection_Resnet50_Final.pth
EOF

run_log="${LOG_ROOT}/sadtalker_smoke_$(date +%Y%m%d_%H%M%S).log"
"${SADTALKER_PYTHON}" inference.py \
  --driven_audio examples/driven_audio/bus_chinese.wav \
  --source_image examples/source_image/full_body_1.png \
  --checkpoint_dir checkpoints \
  --result_dir "${RESULT_ROOT}" \
  --size 256 \
  --batch_size 1 \
  --preprocess crop 2>&1 | tee "${run_log}"

latest_video="$(find "${RESULT_ROOT}" -maxdepth 1 -type f -name '*.mp4' -print | sort | tail -n 1)"
if [[ -z "${latest_video}" ]]; then
  echo "SadTalker finished without producing an MP4." >&2
  exit 7
fi

ffprobe -v error \
  -show_entries format=filename,duration,size \
  -show_entries stream=index,codec_name,width,height,r_frame_rate \
  -of default=noprint_wrappers=1 "${latest_video}" \
  | tee "${LOG_ROOT}/sadtalker_smoke_ffprobe.txt"
sha256sum "${latest_video}" | tee "${LOG_ROOT}/sadtalker_smoke_output_sha256.txt"

echo "GPU smoke test passed: ${latest_video}"
