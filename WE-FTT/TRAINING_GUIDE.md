# 训练脚本使用说明

## 问题诊断和解决方案

### 原始问题
原始的 `scripts/train.py` 无法直接运行，因为存在Python模块导入问题：
- `src/` 目录下的多个文件使用了相对导入（如 `from .config import ...`）
- 这种导入方式在直接运行脚本时会失败，出现 `ImportError: attempted relative import with no known parent package` 错误

### 解决方案
创建了修复版本 `scripts/train_fixed.py`，主要修改：
1. **修改sys.path设置**：将项目根目录添加到Python路径，而不是只添加src目录
2. **更新导入语句**：所有导入都使用绝对路径（`from src.config import ...`）

### 修改的文件对比
```python
# 原始版本 (train.py)
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from config import WEFTTConfig

# 修复版本 (train_fixed.py)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
from src.config import WEFTTConfig
```

---

## 使用方法

### 环境要求
- **Conda环境**: gpytorch (已验证可用)
- **GPU**: 可用4个NVIDIA GeForce RTX 3090
- **Python**: 3.11.7

### 基本命令

#### 1. 训练WE-FTT模型（主模型，带权重增强）
```bash
conda run -n gpytorch python scripts/train_fixed.py \
    --model_name we_ftt \
    --epochs 50 \
    --batch_size 32 \
    --learning_rate 0.001
```

#### 2. 训练基线FT-Transformer（不带权重增强）
```bash
conda run -n gpytorch python scripts/train_fixed.py \
    --model_name ft_transformer \
    --epochs 50 \
    --batch_size 32
```

#### 3. 训练传统机器学习基线模型
```bash
# Random Forest
conda run -n gpytorch python scripts/train_fixed.py --model_name random_forest

# CatBoost
conda run -n gpytorch python scripts/train_fixed.py --model_name catboost

# TabNet
conda run -n gpytorch python scripts/train_fixed.py --model_name tabnet

# XGBoost
conda run -n gpytorch python scripts/train_fixed.py --model_name xgboost

# LightGBM
conda run -n gpytorch python scripts/train_fixed.py --model_name lightgbm
```

### 命令行参数说明

| 参数 | 必需 | 默认值 | 说明 |
|------|------|--------|------|
| `--model_name` | ✓ | - | 模型名称：we_ftt, ft_transformer, random_forest, catboost, tabnet, xgboost, lightgbm |
| `--config_file` | ✗ | None | 配置文件路径（YAML或JSON） |
| `--data_path` | ✗ | 自动 | 训练数据路径 |
| `--output_dir` | ✗ | results | 结果输出目录 |
| `--epochs` | ✗ | 20 | 训练轮数 |
| `--batch_size` | ✗ | 32 | 批次大小 |
| `--learning_rate` | ✗ | 0.001 | 学习率 |
| `--random_seed` | ✗ | 42 | 随机种子 |
| `--gpu` | ✗ | 0 | GPU设备ID |

### 高级用法

#### 使用配置文件
```bash
conda run -n gpytorch python scripts/train_fixed.py \
    --model_name we_ftt \
    --config_file configs/my_config.yaml
```

#### 指定数据路径
```bash
conda run -n gpytorch python scripts/train_fixed.py \
    --model_name we_ftt \
    --data_path /path/to/data.parquet \
    --output_dir my_results
```

#### 使用特定GPU
```bash
conda run -n gpytorch python scripts/train_fixed.py \
    --model_name we_ftt \
    --gpu 1
```

---

## 测试验证

### 快速测试
运行提供的测试脚本来验证所有组件：
```bash
conda run -n gpytorch python test_train_script.py
```

### 预期输出
```
============================================================
测试 train_fixed.py 脚本初始化
============================================================

1. 测试日志设置...
   ✓ 日志设置成功

2. 测试配置加载...
   ✓ WEFTTConfig加载成功
   - 特征列数: 10
   - 权重列数: 10

3. 测试GPU可用性...
   ✓ GPU可用
   - GPU数量: 4
   - 当前GPU: NVIDIA GeForce RTX 3090

4. 测试模型创建...
   ✓ WE-FTT模型创建成功
   - 总参数量: 14,065,922
   - 可训练参数: 14,065,922

5. 测试基线模型配置...
   ✓ BaselineConfig加载成功
```

### 小规模训练测试
进行2个epoch的快速训练测试：
```bash
conda run -n gpytorch python scripts/train_fixed.py \
    --model_name we_ftt \
    --epochs 2 \
    --batch_size 32
```

---

## 输出结果

训练完成后，结果会保存到指定的输出目录（默认为 `results/{model_name}/`）：

```
results/
└── we_ftt/
    ├── results.json          # 训练历史和测试指标
    ├── config.json           # 使用的配置参数
    └── best_model.pth        # 最佳模型权重（深度学习模型）
```

### results.json 内容示例
```json
{
  "train_history": {
    "train_loss": [0.52, 0.41, 0.35, ...],
    "val_loss": [0.48, 0.39, 0.33, ...],
    "val_accuracy": [0.78, 0.82, 0.85, ...]
  },
  "test_results": {
    "accuracy": 0.87,
    "precision": 0.86,
    "recall": 0.88,
    "f1_score": 0.87,
    "cohen_kappa": 0.74,
    "matthews_corrcoef": 0.75
  },
  "best_val_loss": 0.31
}
```

---

## 常见问题

### Q1: 如果仍然遇到导入错误怎么办？
确保从项目根目录运行脚本：
```bash
cd /mnt/hdd_4tb_data/ArchivedWorks/MBT/Final/WE-FTT
conda run -n gpytorch python scripts/train_fixed.py --model_name we_ftt
```

### Q2: 内存不足怎么办？
减小批次大小：
```bash
--batch_size 16  # 或更小
```

### Q3: 如何监控训练进度？
训练日志会实时输出到终端，显示每个epoch的损失和准确率。

### Q4: 如何恢复训练？
目前脚本不支持恢复训练，需要重新训练。可以保存最佳模型后手动加载继续训练。

---

## 注意事项

1. **原始文件保留**：`scripts/train.py` 保持原样未修改，所有修改在 `train_fixed.py`
2. **数据位置**：脚本会自动从 `data/processed/` 目录加载数据
3. **GPU使用**：默认使用GPU 0，可通过 `--gpu` 参数指定其他GPU
4. **早停机制**：深度学习模型默认使用早停（patience=5），防止过拟合
5. **随机种子**：默认seed=42，保证结果可复现

---

## 下一步建议

1. **完整训练**：使用完整数据和50个epoch训练WE-FTT模型
2. **基线对比**：训练所有基线模型进行性能对比
3. **消融研究**：运行 `scripts/run_ablation.py` 进行消融实验
4. **超参数优化**：根据初步结果调整学习率、批次大小等参数

---

## 联系和支持

如果遇到问题：
1. 检查conda环境是否正确激活
2. 验证数据文件是否存在
3. 查看训练日志中的错误信息
4. 运行测试脚本确认环境配置正确
