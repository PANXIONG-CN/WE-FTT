#!/bin/bash
# 快速启动脚本 - 修复版

set -e

echo "=========================================="
echo "WE-FTT 双GPU训练 - 快速启动"
echo "=========================================="

# 激活conda环境
eval "$(conda shell.bash hook)"
conda activate gpytorch

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1
export NCCL_P2P_LEVEL=NVL
export NCCL_IB_DISABLE=1

# 训练参数
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="results/we_ftt_dual_gpu_${TIMESTAMP}"

echo ""
echo "开始训练..."
echo "输出目录: $OUTPUT_DIR"
echo ""

# 运行训练，使用CSV文件
python3 scripts/train_dual_gpu.py \
    --model_name we_ftt \
    --data_path data/processed/downsampled_f0t0.csv \
    --epochs 50 \
    --batch_size 256 \
    --learning_rate 0.001 \
    --weight_decay 0.01 \
    --patience 10 \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "=========================================="
echo "训练完成!"
echo "结果: $OUTPUT_DIR"
echo "=========================================="
