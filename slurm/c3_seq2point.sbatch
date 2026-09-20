#!/bin/bash
#SBATCH -J pslg-c3-s2p
#SBATCH -p RTX3090
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 2-00:00:00
#SBATCH -o /home/%u/pslg_logs/%x-%j.out
#SBATCH -e /home/%u/pslg_logs/%x-%j.err

# C3 Seq2Point first-round validation job (guide §7.13): B0/B1/B2, ratio 0.5,
# test stays locked. Submit only after C2 froze the state library and the
# formal B2 recomposition was rebuilt. One job = one (arm, seed).
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
conda activate pslg-nilm

cd "$HOME/projects/PSLG-NILM"
git rev-parse HEAD
nvidia-smi

# Fill in before submit: frozen commit, manifest, arm and seed for THIS job.
FROZEN_COMMIT=<frozen-commit>
MANIFEST="$HOME/pslg_manifests/c1_nilm_b0_b2_trainval_manifest.json"
ARM=B2           # B0 | B1 | B2
SEED=17          # 17 | 42 | 73
RUN_ROOT="$HOME/pslg_artifacts"
RUN_ID="s2p_${ARM}_r0p5_s${SEED}_${SLURM_JOB_ID}"

python scripts/server_preflight.py \
  --repo-root . --expected-commit "$FROZEN_COMMIT" \
  --manifest "$MANIFEST" \
  --output-root "$RUN_ROOT" --framework torch

python scripts/capture_server_environment.py \
  --repo-root . --output "$RUN_ROOT/${RUN_ID}_environment.json"

python scripts/train_nilm.py \
  --experiment-dir "$HOME/pslg_data/nilm_inputs_frozen_candidate" \
  --arm "$ARM" --seed "$SEED" \
  --batch-size 128 --steps-per-epoch 200 --max-epochs 30 --patience 5 \
  --validation-count 20000 \
  --output-dir "$RUN_ROOT/$RUN_ID"

python scripts/predict_nilm.py \
  --experiment-dir "$HOME/pslg_data/nilm_inputs_frozen_candidate" \
  --checkpoint "$RUN_ROOT/$RUN_ID/best_checkpoint.pt" \
  --arm "$ARM" --partition validation --limit 20000 \
  --output "$RUN_ROOT/$RUN_ID/validation_predictions.npz"

python scripts/evaluate_nilm_predictions.py \
  --predictions "$RUN_ROOT/$RUN_ID/validation_predictions.npz" \
  --output "$RUN_ROOT/$RUN_ID/validation_metrics.json"

python scripts/collect_run_artifacts.py --run-dir "$RUN_ROOT/$RUN_ID"

echo "finished=$(date --iso-8601=seconds)"
