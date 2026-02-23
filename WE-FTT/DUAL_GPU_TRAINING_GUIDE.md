# 双GPU训练指南

## 环境要求

- **硬件**: 2个RTX 3090 GPU（通过NVLink连接）
- **软件**:
  - Conda环境: gpytorch
  - PyTorch 2.1.1 + CUDA 12.1
  - Python 3.11

## 快速开始

### 1. 检查环境

```bash
python3 check_environment.py
```

### 2. 开始训练

**方式一：快速启动（推荐）**
```bash
./quick_start_training.sh
```

**方式二：自定义参数**
```bash
./run_dual_gpu_training.sh
```

**方式三：直接调用Python脚本**
```bash
CUDA_VISIBLE_DEVICES=0,1 python3 scripts/train_dual_gpu.py \
    --model_name we_ftt \
    --data_path data/processed/downsampled_f0t0.csv \
    --epochs 50 \
    --batch_size 256 \
    --learning_rate 0.001 \
    --output_dir results/my_training
```

## 训练参数说明

- `--model_name`: 模型名称（we_ftt 或 ft_transformer）
- `--data_path`: 训练数据路径（CSV或Parquet格式）
- `--epochs`: 训练轮数（默认50）
- `--batch_size`: 每个GPU的批量大小（默认256，总批量=512）
- `--learning_rate`: 学习率（默认0.001）
- `--weight_decay`: 权重衰减（默认0.01）
- `--patience`: 早停的patience（默认10）
- `--output_dir`: 输出目录

## 性能优化

### 批量大小调整

双GPU配置下，总批量大小 = batch_size × 2

推荐设置：
- 小数据集: `--batch_size 128` (总批量256)
- 中等数据集: `--batch_size 256` (总批量512) **[默认]**
- 大数据集: `--batch_size 512` (总批量1024)

每个RTX 3090有23.7GB显存，可以处理较大的批量。

### NVLink优化

脚本已自动设置以下环境变量优化NVLink性能：
```bash
export NCCL_P2P_LEVEL=NVL  # 启用NVLink
export NCCL_IB_DISABLE=1   # 禁用InfiniBand（不适用）
```

## 训练过程监控

### 实时查看日志
```bash
tail -f results/we_ftt_dual_gpu_TIMESTAMP_training.log
```

### GPU使用情况
```bash
watch -n 1 nvidia-smi
```

### 特定GPU监控
```bash
nvidia-smi --query-gpu=index,timestamp,utilization.gpu,utilization.memory,memory.used,memory.free,temperature.gpu,power.draw --format=csv -l 1 -i 0,1
```

## 输出文件

训练完成后，输出目录包含：

```
results/we_ftt_dual_gpu_TIMESTAMP/
├── best_model.pth          # 最佳模型权重（包含完整checkpoint）
├── results.json            # 训练历史和测试结果
└── config.json             # 使用的配置参数
```

### 加载最佳模型

```python
import torch
from src.models.we_ftt import create_we_ftt_model

# 加载checkpoint
checkpoint = torch.load('results/.../best_model.pth')

# 创建模型
model = create_we_ftt_model(
    num_features=10,
    num_classes=2,
    config=checkpoint['config']['model_params'],
    use_weight_enhancement=True
)

# 加载权重
model.load_state_dict(checkpoint['model_state_dict'])

# 查看训练信息
print(f"训练轮数: {checkpoint['epoch']}")
print(f"验证损失: {checkpoint['val_loss']:.4f}")
print(f"验证准确率: {checkpoint['val_accuracy']:.4f}")
```

## 常见问题

### 1. CUDA Out of Memory

降低批量大小：
```bash
python3 scripts/train_dual_gpu.py --batch_size 128 ...
```

### 2. 训练速度慢

检查：
- NVLink是否正常工作
- 数据加载是否成为瓶颈（增加num_workers）
- 混合精度训练已默认启用

### 3. 模型不收敛

尝试：
- 调整学习率: `--learning_rate 0.0005`
- 增加warmup步数（修改脚本）
- 检查数据是否正确归一化

## 高级用法

### 修改模型架构

编辑 `src/config.py` 中的 `WEFTTConfig.BEST_PARAMS`：

```python
BEST_PARAMS = {
    "learning_rate": 0.001,
    "dropout_rate": 0.1,
    "n_heads": 8,        # 注意力头数
    "n_layers": 6,       # Transformer层数
    "hidden_dim": 512,   # 隐藏维度
    ...
}
```

### 使用不同的数据集

```bash
# 使用不同的类型文件
python3 scripts/train_dual_gpu.py \
    --data_path data/processed/downsampled_f1t0.csv \
    ...
```

### 继续训练

当前脚本会保存checkpoint，可以修改代码支持从checkpoint继续训练。

## 性能基准

在双RTX 3090配置下的预期性能：

- **训练速度**: ~1-2秒/epoch（取决于数据大小和批量）
- **显存使用**: 每个GPU约8-12GB（batch_size=256）
- **预期准确率**: ~84%
- **训练时间**: 完整训练约2-5分钟（50 epochs）

## 技术细节

### 分布式训练实现

- 使用PyTorch DistributedDataParallel (DDP)
- NCCL后端用于GPU间通信
- 混合精度训练（torch.cuda.amp）
- 梯度裁剪（max_norm=1.0）
- Cosine学习率调度

### 数据分布

- 使用DistributedSampler确保每个GPU获得不同的数据子集
- 验证和测试集不使用分布式采样
- 支持PIN memory加速GPU传输

## 联系支持

如有问题，请查看：
1. 训练日志文件
2. `check_environment.py` 输出
3. GPU监控信息

调试模式运行：
```bash
NCCL_DEBUG=INFO ./quick_start_training.sh
```
