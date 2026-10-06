#!/bin/bash
# ============================================================================
# LEGACY: 旧项目时期的脚本，包含旧用户目录与旧仓库名，禁止在新集群使用。
# Formal C1/C2/C3 jobs use slurm/c1_gpu_smoke.sbatch, c2_detsec_pc.sbatch and
# c3_seq2point.sbatch instead (guide §4.6 / §5.5).
# ============================================================================
#SBATCH -J detsec_wm
#SBATCH -p RTX3090
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -o slurm_log/detsec_wm-%j.out
#SBATCH -e slurm_log/detsec_wm-%j.err

set -e

cd /home/scnu2024024563/PSLG-NILM
mkdir -p slurm_log

module purge
module load miniconda3/25.5.1-0

source $(conda info --base)/bin/activate
conda activate pslg-nilm

export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL=0
export TF_FORCE_GPU_ALLOW_GROWTH=true
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMBA_NUM_THREADS=1
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cublas/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cufft/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/curand/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cusolver/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cusparse/lib:$LD_LIBRARY_PATH"

echo "HOSTNAME=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "CONDA_PREFIX=$CONDA_PREFIX"
which python
python --version
which nvidia-smi || true
nvidia-smi || true

python -u - <<'PY'
import sys
import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
print("TF:", tf.__version__, flush=True)
print("GPUs:", gpus, flush=True)

if not gpus:
    print("ERROR: TensorFlow cannot see GPU. Stop this job to avoid CPU DETSEC training.", flush=True)
    sys.exit(1)
PY

python -u main.py --config config/config_washing_machine_ukdale_detsec.yaml