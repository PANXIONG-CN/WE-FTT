#!/bin/bash
#
# 快速启动WE-FTT模型训练
#
# 使用方法:
#   bash quick_start_training.sh

echo "=========================================="
echo "WE-FTT Model Training - Quick Start"
echo "=========================================="

# 配置参数
MODEL_NAME="we_ftt"
DATA_PATH="/mnt/hdd_4tb_data/ArchivedWorks/MBT/Final/WE-FTT/data/processed/training_data_labeled.parquet"
EPOCHS=50
BATCH_SIZE=10000
LEARNING_RATE=0.001
OUTPUT_DIR="results"
GPU_ID=0
LOG_FILE="training_log.txt"

echo ""
echo "Training Configuration:"
echo "  Model: $MODEL_NAME"
echo "  Data: training_data_labeled.parquet (346M samples)"
echo "  Epochs: $EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Learning Rate: $LEARNING_RATE"
echo "  GPU: $GPU_ID"
echo "  Output: $OUTPUT_DIR/$MODEL_NAME/"
echo ""

# 检查数据文件是否存在
if [ ! -f "$DATA_PATH" ]; then
    echo "ERROR: Data file not found: $DATA_PATH"
    exit 1
fi

echo "Starting training..."
echo "Log file: $LOG_FILE"
echo ""

# 启动训练
conda run -n gpytorch python scripts/train_with_save.py \
    --model_name "$MODEL_NAME" \
    --data_path "$DATA_PATH" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE" \
    --output_dir "$OUTPUT_DIR" \
    --gpu "$GPU_ID" \
    2>&1 | tee "$LOG_FILE"

# 检查训练是否成功
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Training completed successfully!"
    echo "=========================================="
    echo "Results saved to: $OUTPUT_DIR/$MODEL_NAME/"
    echo "  - best_model.pth: Model checkpoint"
    echo "  - results.json: Training metrics"
    echo "  - config.json: Configuration"
else
    echo ""
    echo "=========================================="
    echo "Training failed! Check $LOG_FILE for details."
    echo "=========================================="
    exit 1
fi
