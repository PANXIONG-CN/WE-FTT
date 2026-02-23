#!/bin/bash
#
# 自动监控训练进度 - 每10分钟检查一次
# 日志保存到 monitoring_log.txt
#

LOG_FILE="monitoring_log.txt"
CHECK_INTERVAL=600  # 10分钟 = 600秒

echo "=========================================="
echo "WE-FTT 自动训练监控系统"
echo "=========================================="
echo "检查间隔: 10分钟"
echo "日志文件: $LOG_FILE"
echo "启动时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# 写入初始日志
{
    echo "=========================================="
    echo "监控开始: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=========================================="
    echo ""
} >> $LOG_FILE

# 检查函数
check_training() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    # 查找训练进程
    TRAIN_PID=$(ps aux | grep "train_with_save.py.*we_ftt" | grep -v grep | awk '{print $2}' | head -1)
    
    {
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "检查时间: $timestamp"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        
        if [ -z "$TRAIN_PID" ]; then
            echo "⚠️  训练进程未运行！"
            echo ""
            
            # 检查是否正常完成
            if [ -f "results/we_ftt/best_model.pth" ]; then
                echo "✓ 训练已完成！模型已保存。"
                echo ""
                echo "结果文件："
                ls -lh results/we_ftt/ 2>/dev/null | tail -n +2
                echo ""
                
                # 显示最终结果
                if [ -f "results/we_ftt/results.json" ]; then
                    echo "训练结果摘要："
                    python3 -c "
import json
try:
    with open('results/we_ftt/results.json', 'r') as f:
        results = json.load(f)
    if 'test_results' in results:
        print('  测试准确率: {:.4f}'.format(results['test_results'].get('accuracy', 0)))
        print('  F1分数: {:.4f}'.format(results['test_results'].get('f1_score', 0)))
except: pass
" 2>/dev/null
                fi
            else
                echo "❌ 训练进程异常退出！"
                echo ""
                echo "最后的日志输出："
                tail -20 training_log.txt 2>/dev/null
            fi
            
            echo ""
            echo "监控将在下次检查后自动停止。"
            return 1
        fi
        
        # 进程信息
        echo "✓ 训练进程运行中 (PID: $TRAIN_PID)"
        echo ""
        
        # CPU和内存
        echo "📊 资源使用:"
        ps aux | grep "^[^ ]* *$TRAIN_PID " | awk '{
            printf "  CPU: %s%%  |  内存: %s%% (%d MB)\n", 
            $3, $4, int($6/1024)
        }'
        
        # 运行时长
        START_TIME=$(ps -p $TRAIN_PID -o lstart= 2>/dev/null)
        if [ -n "$START_TIME" ]; then
            echo "  启动时间: $START_TIME"
        fi
        
        # GPU状态
        echo ""
        echo "🎮 GPU状态:"
        nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader 2>/dev/null | head -1 | \
            awk -F', ' '{printf "  GPU %s: %s利用率, %s/%s显存, %s°C\n", $1, $2, $3, $4, $5}'
        
        # 训练日志最新输出
        echo ""
        echo "📝 最新训练日志:"
        if [ -s training_log.txt ]; then
            LAST_LINES=$(tail -20 training_log.txt 2>/dev/null | grep -E "Epoch|Loss|Accuracy|Starting|Preparing" | tail -5)
            if [ -n "$LAST_LINES" ]; then
                echo "$LAST_LINES" | sed 's/^/  /'
            else
                TOTAL_LINES=$(wc -l < training_log.txt 2>/dev/null)
                echo "  日志文件已有 $TOTAL_LINES 行"
                echo "  最新内容:"
                tail -3 training_log.txt 2>/dev/null | sed 's/^/    /'
            fi
        else
            echo "  等待日志输出... (数据加载阶段)"
        fi
        
        # 输出文件检查
        echo ""
        echo "💾 输出文件:"
        if [ -d "results/we_ftt" ]; then
            ls -lh results/we_ftt/ 2>/dev/null | tail -n +2 | awk '{printf "  %-20s %8s  %s %s\n", $9, $5, $6, $7}'
        else
            echo "  结果目录尚未创建 (训练未开始)"
        fi
        
        echo ""
        
    } >> $LOG_FILE
    
    return 0
}

# 主循环
echo "开始监控... (按 Ctrl+C 停止)"
echo ""

check_count=0
while true; do
    check_count=$((check_count + 1))
    echo "[检查 #$check_count] $(date '+%H:%M:%S')"
    
    # 执行检查
    if ! check_training; then
        echo "训练已结束，监控停止。"
        {
            echo "监控结束: $(date '+%Y-%m-%d %H:%M:%S')"
            echo "=========================================="
            echo ""
        } >> $LOG_FILE
        break
    fi
    
    echo "下次检查: $(date -d "+10 minutes" '+%H:%M:%S')"
    echo ""
    
    # 等待10分钟
    sleep $CHECK_INTERVAL
done

echo ""
echo "监控日志已保存到: $LOG_FILE"
echo "查看完整日志: cat $LOG_FILE"
