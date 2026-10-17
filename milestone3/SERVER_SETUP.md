# Two-stage server setup for the M3 extension

This setup keeps paid GPU time short. Configure and test the public M3 pipeline
on a CPU instance first, then attach or start a GPU only for batch video
generation. Do not place account tokens, private videos or SSH keys in Git.

## Stage 1: CPU-only preparation

Use Ubuntu 22.04 or 24.04 with at least 4 vCPU, 16 GB RAM and 50 GB persistent
disk. Clone the repository, then run:

```bash
git clone https://github.com/zyan0246-del/FIT5230-Light-Context-Matters.git
cd FIT5230-Light-Context-Matters/milestone3
chmod +x scripts/bootstrap_server_cpu.sh scripts/check_server_gpu.sh
./scripts/bootstrap_server_cpu.sh
```

If the provider image already contains a verified CUDA-enabled PyTorch build,
reuse it rather than downloading an unpinned replacement through Whisper:

```bash
export M3_PYTHON_BIN=/path/to/provider/python
export M3_USE_SYSTEM_SITE_PACKAGES=1
./scripts/bootstrap_server_cpu.sh
```

Set `M3_SKIP_APT=1` only after the system packages have completed once.

### AutoDL no-GPU mode

Choose the final NVIDIA/CUDA image before installing anything, with Python 3.10
available. Keep the verified M3 environment separate from the later generator
environment. Put the repository, virtual environment and data under AutoDL's
persistent data path rather than relying on the system disk:

```bash
cd /root/autodl-tmp
git clone https://github.com/zyan0246-del/FIT5230-Light-Context-Matters.git
cd FIT5230-Light-Context-Matters/milestone3
chmod +x scripts/bootstrap_server_cpu.sh scripts/check_server_gpu.sh
export M3_VENV_PATH=/root/autodl-tmp/m3-venv
export M3_DATA_ROOT=/root/autodl-tmp/m3-data
export M3_LOG_ROOT=/root/autodl-tmp/m3-logs
./scripts/bootstrap_server_cpu.sh
```

Use the same three environment variables after every login. Saving them in a
local shell profile is acceptable, but never place credentials there. Normal
shutdown preserves the instance, while changing/resetting an image may remove
the system disk; keep a separate copy of irreplaceable inputs and results.

The bootstrap installs FFmpeg and Python, creates `.venv`, installs the exact M3
requirements, runs the smoke pipeline and the regression tests, and creates:

```text
server_data/
  raw_real/
  generated_fake/
  processed/
  manifests/
  cache/
server_logs/
```

Both directories are ignored by Git. To put data on a mounted persistent disk,
set paths before running the bootstrap:

```bash
export M3_DATA_ROOT=/workspace/m3-data
export M3_LOG_ROOT=/workspace/m3-logs
./scripts/bootstrap_server_cpu.sh
```

Optional public-pilot download:

```bash
M3_DOWNLOAD_PUBLIC=1 ./scripts/bootstrap_server_cpu.sh
```

The successful CPU gate is:

- `python -m s2f.cli smoke` completes;
- all regression tests pass;
- `server_logs/cpu_environment.txt` records the commit and environment;
- `ffmpeg -version` is available.

## Stage 2: GPU activation

Start from an NVIDIA driver/CUDA image supplied by the server provider. Reuse
the persistent data disk and repository checkout, then run:

```bash
cd FIT5230-Light-Context-Matters/milestone3
./scripts/check_server_gpu.sh
```

For the prepared GPUFree/SadTalker environment, first confirm that the paid or
dedicated GPU instance has started, then run the guarded one-video test:

```bash
cd /root/gpufree-data/FIT5230-M3-Server/milestone3
M3_GPU_MODE_CONFIRMED=1 ./scripts/run_sadtalker_smoke_gpu.sh
```

The script refuses to generate by default, requires at least 6000 MiB of free
GPU memory, verifies all four model files by SHA-256, generates one official
example at 256 px without an enhancer, and records `nvidia-smi`, `ffprobe` and
output-hash evidence under `/root/gpufree-data/m3-logs`.

Do not install an arbitrary CUDA toolkit during the CPU stage. PyTorch and the
talking-face generator must be pinned to the selected GPU image in a separate
environment so they cannot break the already verified M3 environment.

## Before paying for GPU time

Prepare these items while the CPU instance is running:

1. an approved list of real source clips and their licences/provenance;
2. a frozen manifest with clip, speaker, source and split IDs;
3. one portrait/source frame and one audio file per generated clip;
4. fixed output settings: 25 FPS, H.264, resolution and audio sample rate;
5. a new hold-out group that is not viewed during model selection.

The GPU should be stopped after batch generation. Feature extraction, Ridge,
logistic regression and evaluation remain CPU tasks.
