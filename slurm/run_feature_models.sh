#!/bin/bash
# ============================================================================
# LEGACY: 旧项目时期的脚本，包含旧用户目录与旧仓库名，禁止在新集群使用。
# Formal C1/C2/C3 jobs use slurm/c1_gpu_smoke.sbatch, c2_detsec_pc.sbatch and
# c3_seq2point.sbatch instead (guide §4.6 / §5.5).
# ============================================================================
#SBATCH -J PSLG-FEAT-MODELS
#SBATCH -p RTX3090
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -o /home/scnu2023024258/data/code/PSLG-NILM-ADVANCED/slurm/slurm_log/job-%x-%j.out
#SBATCH -e /home/scnu2023024258/data/code/PSLG-NILM-ADVANCED/slurm/slurm_log/job-%x-%j.err
#SBATCH --time=72:00:00

# 特征模型网格：同一切分方法下遍历所有特征提取模型。
# 特征提取有内容寻址缓存：重复提交同一 (模型, 超参, 输入) 组合会直接命中，
# 只有真正变化的部分才重训。所有选择走 CLI 参数，不再用 sed 改 config。
source "$(dirname "$0")/env.sh"

APPLIANCE="${APPLIANCE:-fridge}"
SEGMENT_METHOD="${SEGMENT_METHOD:-clasp}"
MODELS=("detsec" "bilstm_ae" "bilstm_ae_attention" "autoencoder")
if [ -n "${MODELS_ENV:-}" ]; then read -r -a MODELS <<< "$MODELS_ENV"; fi

for model in "${MODELS[@]}"; do
    echo "=================================================="
    echo "[grid] feature_model=$model  segment_method=$SEGMENT_METHOD"
    echo "=================================================="
    python main.py \
        --appliance "$APPLIANCE" \
        --steps extract,segment,feature \
        --segment-method "$SEGMENT_METHOD" \
        --feature-model "$model"
done

echo "Job finished on: $(date)"
