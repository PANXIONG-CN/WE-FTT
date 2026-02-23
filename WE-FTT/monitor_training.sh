#!/bin/bash
#
# 实时监控训练进度
#

echo "=========================================="
echo "WE-FTT Training Monitor"
echo "=========================================="
echo ""

# 训练进程PID
TRAIN_PID=$(ps aux | grep "train_with_save.py" | grep -v grep | awk '{print $2}' | head -1)

if [ -z "$TRAIN_PID" ]; then
    echo "❌ 训练进程未运行！"
    exit 1
fi

echo "✓ 训练进程运行中 (PID: $TRAIN_PID)"
echo ""

# 循环监控
while true; do
    clear
    echo "=========================================="
    echo "WE-FTT Training Monitor - $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=========================================="
    echo ""
    
    # 检查进程是否还在运行
    if ! ps -p $TRAIN_PID > /dev/null 2>&1; then
        echo "❌ 训练进程已结束！"
        echo ""
        echo "查看结果："
        echo "  tail -100 training_log.txt"
        echo "  ls -lh results/we_ftt/"
        break
    fi
    
    # 进程信息
    echo "📊 进程状态:"
    ps aux | grep $TRAIN_PID | grep -v grep | awk '{printf "  CPU: %s%%  |  内存: %s%%  |  运行时间: %s\n", $3, $4, $10}'
    
    # 内存详情
    MEM_MB=$(ps -p $TRAIN_PID -o rss= | awk '{print int($1/1024)}')
    echo "  内存使用: ${MEM_MB} MB"
    
    # GPU状态
    echo ""
    echo "🎮 GPU状态:"
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader | head -1 | \
        awk -F', ' '{printf "  GPU %s (%s): %s利用率, %s / %s 显存\n", $1, $2, $3, $4, $5}'
    
    # 日志文件
    echo ""
    echo "📝 训练日志:"
    if [ -s training_log.txt ]; then
        echo "  最新10行:"
        tail -10 training_log.txt | sed 's/^/    /'
    else
        echo "  等待日志输出... (数据加载中)"
    fi
    
    # 输出文件
    echo ""
    echo "💾 输出文件:"
    if [ -d "results/we_ftt" ]; then
        ls -lh results/we_ftt/ 2>/dev/null | tail -n +2 | awk '{printf "  %s  %s  %s\n", $9, $5, $6" "$7}'
    else
        echo "  结果目录尚未创建"
    fi
    
    echo ""
    echo "=========================================="
    echo "按 Ctrl+C 退出监控（不会停止训练）"
    echo "=========================================="
    
    # 等待5秒后刷新
    sleep 5
done
