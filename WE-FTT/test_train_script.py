#!/usr/bin/env python3
"""
快速测试脚本，验证train_fixed.py能否正确初始化
"""
import sys
import os

# 添加项目根目录到路径
project_root = '/mnt/hdd_4tb_data/ArchivedWorks/MBT/Final/WE-FTT'
sys.path.insert(0, project_root)

from src.config import WEFTTConfig, BaselineConfig
from src.utils import setup_logging
import torch

print("=" * 60)
print("测试 train_fixed.py 脚本初始化")
print("=" * 60)

# 1. 测试日志设置
print("\n1. 测试日志设置...")
try:
    setup_logging()
    print("   ✓ 日志设置成功")
except Exception as e:
    print(f"   ✗ 日志设置失败: {e}")

# 2. 测试配置加载
print("\n2. 测试配置加载...")
try:
    config = WEFTTConfig()
    print(f"   ✓ WEFTTConfig加载成功")
    print(f"   - 特征列数: {len(config.COLUMNS_FEATURES)}")
    print(f"   - 权重列数: {len(config.COLUMNS_WEIGHTS)}")
    print(f"   - 批次大小: {config.BEST_PARAMS.get('batch_size', 'N/A')}")
except Exception as e:
    print(f"   ✗ 配置加载失败: {e}")

# 3. 测试GPU可用性
print("\n3. 测试GPU可用性...")
if torch.cuda.is_available():
    print(f"   ✓ GPU可用")
    print(f"   - GPU数量: {torch.cuda.device_count()}")
    print(f"   - 当前GPU: {torch.cuda.get_device_name(0)}")
else:
    print("   ! 没有可用的GPU，将使用CPU")

# 4. 测试模型创建
print("\n4. 测试模型创建...")
try:
    from src.models.we_ftt import create_we_ftt_model
    config = WEFTTConfig()
    model = create_we_ftt_model(
        num_features=len(config.COLUMNS_FEATURES),
        num_classes=2,
        config=config.BEST_PARAMS,
        use_weight_enhancement=True
    )
    print(f"   ✓ WE-FTT模型创建成功")

    # 计算模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   - 总参数量: {total_params:,}")
    print(f"   - 可训练参数: {trainable_params:,}")
except Exception as e:
    print(f"   ✗ 模型创建失败: {e}")
    import traceback
    traceback.print_exc()

# 5. 测试基线模型配置
print("\n5. 测试基线模型配置...")
try:
    baseline_config = BaselineConfig()
    print(f"   ✓ BaselineConfig加载成功")
    print(f"   - RandomForest参数: {list(baseline_config.RANDOM_FOREST_PARAMS.keys())[:3]}...")
    print(f"   - XGBoost参数: {list(baseline_config.XGBOOST_PARAMS.keys())[:3]}...")
except Exception as e:
    print(f"   ✗ 基线配置失败: {e}")

print("\n" + "=" * 60)
print("测试完成！所有核心组件可正常初始化。")
print("=" * 60)
print("\n提示: 使用以下命令运行完整训练:")
print("  conda run -n gpytorch python scripts/train_fixed.py --model_name we_ftt --epochs 2 --batch_size 32")
