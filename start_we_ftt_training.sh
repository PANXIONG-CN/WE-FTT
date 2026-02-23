#!/bin/bash

# WE-FTT训练启动脚本
# 生成时间: 2025-11-04

echo "=========================================="
echo "启动WE-FTT模型训练"
echo "=========================================="

# 激活conda环境
source /home/panxiong/miniconda3/etc/profile.d/conda.sh
conda activate gpytorch

# 进入工作目录
cd /mnt/hdd_4tb_data/ArchivedWorks/MBT/Final

# 设置日志文件名
LOG_FILE="we_ftt_train_$(date +%Y%m%d_%H%M%S).log"

echo "训练脚本: WE_FT_Transformer_train.py"
echo "日志文件: $LOG_FILE"
echo "数据文件: /mnt/hdd_4tb_data/ArchivedWorks/MBT/FTT/Mindformers/training_dataset_demo1.parquet"
echo ""

# 启动训练
nohup python WE_FT_Transformer_train.py > "$LOG_FILE" 2>&1 &

# 保存进程ID
TRAIN_PID=$!
echo "训练进程ID: $TRAIN_PID"
echo "$TRAIN_PID" > we_ftt_train.pid

echo ""
echo "=========================================="
echo "训练已在后台启动！"
echo "=========================================="
echo ""
echo "监控训练进度："
echo "  tail -f $LOG_FILE"
echo ""
echo "检查进程状态："
echo "  ps aux | grep $TRAIN_PID"
echo ""
echo "停止训练："
echo "  kill $TRAIN_PID"
echo ""

# 等待2秒，显示初始日志
sleep 3
echo "初始日志："
tail -30 "$LOG_FILE"
