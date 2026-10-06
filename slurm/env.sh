#!/bin/bash
# ============================================================================
# LEGACY / DO-NOT-USE for C1+ formal jobs (guide §4.6).
# This preamble targets the OLD project conda env "PSLG-NILM" and the old
# module alias cuda-toolkit/12.1. It is kept only because the legacy
# run_*.sh wrappers source it. Formal C1/C2/C3 jobs must NOT source this
# file; they use environment_detsec_server.yml / environment_nilm_server.yml
# and their own module handling in the sbatch templates.
# ============================================================================
# Common environment preamble for all PSLG-NILM-ADVANCED slurm jobs.
# Sourced by the run_*.sh scripts — do not submit this file directly.

# --- 1. 环境准备 ---
# 不使用 module purge，确保驱动相关的基础环境不被清理
module load cuda-toolkit/12.1

export PYTHONNOUSERSITE=1 # 禁止使用用户目录下的 Python 包
export PYTHONPATH=""      # 清空 Python 路径
export PYTHONUNBUFFERED=1 # 实时打印日志

# --- 2. 激活 Conda 环境 ---
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/bin/activate"
conda activate PSLG-NILM

# --- 3. 修复动态库搜索路径 ---
# 优先级：Conda NVIDIA库 > Conda基础库 > 系统路径(含驱动)
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cudnn/lib:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# --- 4. 诊断信息 ---
echo "Job started on: $(date)"
echo "Running on node: $(hostname)"
echo "Allocated GPU: $CUDA_VISIBLE_DEVICES"
nvidia-smi

# --- 5. 项目目录 ---
PROJECT_DIR="/home/scnu2023024258/data/code/PSLG-NILM-ADVANCED"
cd "$PROJECT_DIR" || exit 1
