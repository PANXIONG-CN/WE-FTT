#!/bin/bash
# 双GPU训练启动脚本

set -e  # 遇到错误立即退出

echo "========================================"
echo "WE-FTT 双GPU分布式训练"
echo "========================================"
echo ""

# 激活conda环境
CONDA_ENV="gpytorch"  # 可以根据需要修改

echo "激活Conda环境: $CONDA_ENV"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate $CONDA_ENV

# 检查PyTorch和CUDA
echo ""
echo "检查环境..."
python3 -c "import torch; print(f'PyTorch版本: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'CUDA版本: {torch.version.cuda}'); print(f'可用GPU数量: {torch.cuda.device_count()}')"

echo ""
echo "GPU信息:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv

echo ""
echo "========================================"
echo "开始训练"
echo "========================================"

# 设置只使用GPU 0和1（两个RTX 3090）
export CUDA_VISIBLE_DEVICES=0,1

# 设置NCCL参数以优化NVLink性能
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=NVL  # 启用NVLink

# 训练参数（可以根据需要调整）
MODEL_NAME="we_ftt"
EPOCHS=50
BATCH_SIZE=256  # 每个GPU的批量大小，总批量=256*2=512
LEARNING_RATE=0.001
PATIENCE=10
OUTPUT_DIR="results/dual_gpu_training_$(date +%Y%m%d_%H%M%S)"

echo ""
echo "训练参数:"
echo "  模型: $MODEL_NAME"
echo "  轮数: $EPOCHS"
echo "  批量大小(每GPU): $BATCH_SIZE"
echo "  总批量大小: $((BATCH_SIZE * 2))"
echo "  学习率: $LEARNING_RATE"
echo "  早停patience: $PATIENCE"
echo "  输出目录: $OUTPUT_DIR"
echo ""

# 运行训练
python3 scripts/train_dual_gpu.py \
    --model_name $MODEL_NAME \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --patience $PATIENCE \
    --output_dir $OUTPUT_DIR

echo ""
echo "========================================"
echo "训练完成!"
echo "========================================"
echo "结果保存在: $OUTPUT_DIR"
echo "最佳模型: $OUTPUT_DIR/best_model.pth"
