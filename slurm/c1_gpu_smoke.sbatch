#!/bin/bash
#SBATCH -J pslg-c1-gpu-smoke
#SBATCH -p RTX3090
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 00:40:00
#SBATCH -o /home/%u/pslg_logs/%x-%j.out
#SBATCH -e /home/%u/pslg_logs/%x-%j.err

# C1 GPU framework smoke + environment capture (guide §7.11 steps 4-5).
# Submit only after the data manifests verified OK on the server.
set -euo pipefail

echo "started=$(date --iso-8601=seconds)"
echo "job_id=$SLURM_JOB_ID"
echo "host=$(hostname)"
echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"

module purge
module load miniconda3/25.5.1-0
module load cuda-toolkit/12.1.1
module list

ARTIFACT_ROOT="${HOME}/pslg_artifacts/c1_gpu_smoke_${SLURM_JOB_ID}"
mkdir -p "$ARTIFACT_ROOT"

source "$(conda info --base)/bin/activate"
conda activate pslg-detsec
cd "$HOME/projects/PSLG-NILM"
git rev-parse HEAD
nvidia-smi

python scripts/capture_server_environment.py \
  --repo-root . --output "$ARTIFACT_ROOT/environment_detsec.json"

python scripts/gpu_framework_smoke.py --framework tensorflow \
  --output "$ARTIFACT_ROOT/gpu_smoke_tensorflow.json"

conda deactivate
conda activate pslg-nilm

python scripts/capture_server_environment.py \
  --repo-root . --output "$ARTIFACT_ROOT/environment_nilm.json"

python scripts/gpu_framework_smoke.py --framework torch \
  --output "$ARTIFACT_ROOT/gpu_smoke_torch.json"

echo "finished=$(date --iso-8601=seconds)"
