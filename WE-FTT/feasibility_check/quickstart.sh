#!/bin/bash
# 快速启动脚本 - 可行性验证
# 使用方法: bash quickstart.sh

set -e  # 遇到错误立即退出

echo "========================================"
echo "可行性验证 - 快速启动"
echo "========================================"
echo ""

# 检查Python版本
echo "检查Python版本..."
python --version
echo ""

# 检查当前目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "工作目录: $SCRIPT_DIR"
cd "$SCRIPT_DIR"
echo ""

# 检查必需的脚本文件
echo "检查脚本文件..."
REQUIRED_FILES=(
    "step1_model_verification.py"
    "step2_data_audit.py"
    "step3_end_to_end.py"
    "run_all_checks.py"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo "❌ 缺少文件: $file"
        exit 1
    fi
    echo "✅ $file"
done
echo ""

# 询问用户是运行全部还是单步
echo "请选择运行模式:"
echo "  1) 运行全部验证（推荐，约3.5小时）"
echo "  2) 只运行Step 1 - 模型验证（约30分钟）"
echo "  3) 只运行Step 2 - 数据审计（约1小时）"
echo "  4) 只运行Step 3 - 端到端验证（约2小时）"
echo ""
read -p "请输入选择 [1-4]: " choice

case $choice in
    1)
        echo ""
        echo "🚀 运行全部验证..."
        python run_all_checks.py
        ;;
    2)
        echo ""
        echo "🚀 运行Step 1 - 模型验证..."
        python step1_model_verification.py
        echo ""
        echo "📄 查看报告: cat step1_report.txt"
        ;;
    3)
        echo ""
        echo "🚀 运行Step 2 - 数据审计..."
        python step2_data_audit.py
        echo ""
        echo "📄 查看报告: cat step2_report.txt"
        ;;
    4)
        echo ""
        echo "🚀 运行Step 3 - 端到端验证..."
        python step3_end_to_end.py
        echo ""
        echo "📄 查看报告: cat step3_report.txt"
        ;;
    *)
        echo "❌ 无效选择，退出"
        exit 1
        ;;
esac

echo ""
echo "========================================"
echo "✅ 验证完成"
echo "========================================"
echo ""
echo "生成的报告文件:"
ls -lh *.txt 2>/dev/null || echo "暂无报告文件"
echo ""
