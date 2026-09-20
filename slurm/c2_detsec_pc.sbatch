#!/bin/bash
#SBATCH -J pslg-c2-detsecpc
#SBATCH -p A6000
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 1-00:00:00
#SBATCH -o /home/%u/pslg_logs/%x-%j.out
#SBATCH -e /home/%u/pslg_logs/%x-%j.err

# C2 formal DETSEC-PC state discovery (guide §7.12): train-only, k=3/4/5.
# Submit only after C1 passed. The preflight gate verifies the frozen commit,
# the state-discovery manifest and framework imports; any failure aborts.
set -euo pipefail

echo "started=$(date --iso-8601=seconds)"
echo "job_id=$SLURM_JOB_ID"
echo "host=$(hostname)"
echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"

module purge
module load miniconda3/25.5.1-0
module load cuda-toolkit/12.1.1
module list

source "$(conda info --base)/bin/activate"
conda activate pslg-detsec

cd "$HOME/projects/PSLG-NILM"
git rev-parse HEAD
nvidia-smi

FROZEN_COMMIT=<frozen-commit>
MANIFEST="$HOME/pslg_manifests/c1_state_discovery_trainonly_manifest.json"
RUN_ROOT="$HOME/pslg_artifacts"

python scripts/server_preflight.py \
  --repo-root . --expected-commit "$FROZEN_COMMIT" \
  --manifest "$MANIFEST" \
  --output-root "$RUN_ROOT" --framework tensorflow

python scripts/capture_server_environment.py \
  --repo-root . --output "$RUN_ROOT/c2_detsec_${SLURM_JOB_ID}_environment.json"

python scripts/train_state_discovery.py \
  --config config/experiments/core_wm_state_discovery_detsec_pc.yaml \
  --output-root "$RUN_ROOT/state_discovery_detsec_pc_${SLURM_JOB_ID}" \
  --run-id "detsecpc_k345_trainonly_${SLURM_JOB_ID}" \
  --k 3,4,5 --deterministic-tf

echo "finished=$(date --iso-8601=seconds)"
