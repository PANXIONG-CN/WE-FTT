# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

这是一个基于PyTorch的机器学习研究项目，用于利用微波亮度温度（MBT）数据进行地震前兆检测。该项目实现了权重增强特征Transformer（WE-FTT），它将关联规则挖掘与深度学习相结合，以识别主要地震（M≥7.0）的环境特定前兆。

## 核心架构

### 主要模型实现
- **src/models/we_ftt.py**: 主要的权重增强特征Transformer实现，包含知识引导的特征加权机制
  - `WEFTTransformerModel`: 核心WE-FTT模型类，集成了权重增强功能
  - `WeightFusionLayer`: 权重融合层，使用门控机制融合特征和权重嵌入
  - `ClassificationHead`: 分类头，支持多种激活函数和归一化方法
- **src/models/components.py**: 可重用的模型组件（TransformerBlock, FeatureEmbedding等）
- **src/models/baselines.py**: 基线模型实现（RandomForest, CatBoost, TabNet, XGBoost, LightGBM）

### 数据挖掘组件
- **src/association_mining.py**: 知识挖掘核心模块
  - `ClusterAnalyzer`: K-means聚类分析器
  - `AprioriMiner`: Apriori关联规则挖掘器
  - `WeightCalculator`: 特征权重计算器
  - `KnowledgeMiner`: 整合的知识挖掘流程

### 配置管理
- **src/config.py**: 集中式配置管理
  - `Config`: 基础配置（数据路径、特征列、训练参数）
  - `WEFTTConfig`: WE-FTT模型最佳超参数配置
  - `BaselineConfig`: 基线模型参数配置
  - `DataProcessingConfig`: 数据处理和挖掘配置

## 关键数据结构

### 输入特征
所有模型使用AMSR-2亮度温度数据，跨越多个频率和极化：
- 特征列：BT_06_H/V, BT_10_H/V, BT_23_H/V, BT_36_H/V, BT_89_H/V（共10个特征）
- 权重列：对应的`{feature}_cluster_labels_weight`列

### 知识集成
WE-FTT模型将挖掘的关联规则作为特征权重集成：
- 权重通过离散化MBT数据的频繁项集挖掘得出
- 使用门控机制动态融合特征嵌入和权重嵌入

### 数据组织
- **data/raw/**: 原始数据（地震目录、MBT数据）
- **data/processed/**: 处理后的parquet文件（downsampled_f{0,1}t{0-4}.parquet）
- **results/**: 训练结果和实验输出（从git中排除）

## 常用开发命令

### 训练模型
```bash
# 训练权重增强FT-Transformer（主模型）
python scripts/train.py --model_name we_ftt --epochs 50 --batch_size 32 --learning_rate 0.001

# 训练基线FT-Transformer（无权重增强）
python scripts/train.py --model_name ft_transformer --epochs 50

# 训练基线模型
python scripts/train.py --model_name random_forest
python scripts/train.py --model_name catboost
python scripts/train.py --model_name tabnet
python scripts/train.py --model_name xgboost
python scripts/train.py --model_name lightgbm
```

### 运行消融研究
```bash
# 运行完整的消融研究（20+个实验）
python scripts/run_ablation.py --output_dir results/ablation_study

# 并行运行消融实验
python scripts/run_ablation.py --parallel --num_epochs 20

# 运行特定的消融实验
python scripts/run_ablation.py --experiments no_weight_enhancement gelu_activation
```

### 数据处理和知识挖掘
```bash
# 运行数据预处理流程
python scripts/run_preprocessing.py --input_data data/raw/mbt_data.parquet --output_dir data/processed

# 运行关联规则挖掘（独立）
python -c "from src.association_mining import run_knowledge_mining; run_knowledge_mining('data/processed/downsampled_f0t0.parquet', ['BT_06_H', 'BT_06_V', ...])"
```

### 可行性检查（数据验证）
```bash
# 运行所有可行性检查
cd feasibility_check
python run_all_checks.py

# 运行单个检查步骤
python step1_model_verification.py  # 模型架构验证
python step2_data_audit.py          # 数据完整性审计
python step3_end_to_end.py          # 端到端测试
```

## 训练流程架构

1. **数据加载**: `DatasetCreator`类从parquet文件加载数据并创建训练/验证/测试集
2. **知识挖掘**:
   - K-means聚类离散化特征
   - Apriori算法挖掘频繁项集和关联规则
   - 基于规则计算特征权重
3. **模型训练**:
   - `ModelTrainer`类统一处理所有模型的训练
   - 支持PyTorch深度学习模型和sklearn传统模型
   - 实现早停、学习率调度、混合精度训练
4. **评估**: 计算多种指标（准确率、F1、MCC、Cohen's Kappa、ROC-AUC）

## 模型配置系统

模型使用基于类的配置系统（src/config.py）：
- 超参数定义在各个Config类中
- 关键设置：batch_size=100000, epochs=20, learning_rate=0.001
- 数据路径使用相对于项目根目录的Path对象
- 可通过命令行参数或配置文件覆盖默认设置

## 消融研究体系

消融研究测试以下组件的贡献：
- 权重增强机制（核心创新）
- 位置编码（固定vs可学习）
- 注意力头数量
- 激活函数（ReLU, GELU, Swish）
- 损失函数（交叉熵, Focal Loss）
- 归一化方法（BatchNorm, LayerNorm）
- 融合策略（门控, 加法, 乘法）
- 模型大小（层数、隐藏维度）
- Dropout率
- 学习率调度策略

## 重要实现细节

### 权重增强机制
- 特征和权重分别通过线性层投影到模型维度
- 使用门控融合层动态组合特征和权重嵌入
- 门控权重基于拼接的特征-权重表示计算
- 包含残差连接和层归一化

### 动态Focal Loss
- 实现自适应gamma参数，根据训练进度调整
- 处理类别不平衡的地震数据
- 支持alpha权重用于类别重新加权

### 分布式训练支持
- 模型支持DistributedDataParallel
- 使用GradScaler进行混合精度训练
- 内存效率优化用于大批量大小（100k+样本）

## 环境要求

- **Python**: 3.9+ (推荐3.11)
- **PyTorch**: 1.13+ with CUDA 11.7+
- **关键依赖**: scikit-learn, pandas, numpy, optuna, matplotlib
- **硬件**: 推荐GPU（CUDA），至少8GB RAM
- **数据格式**: Parquet文件用于高效大规模数据处理

## 结果和输出

- 模型训练输出保存到`results/{model_name}/`
- 消融研究结果在`results/ablation_study/`
- 每个实验包含：
  - results.json: 训练历史和测试指标
  - config.json: 使用的超参数
  - model.pth: 保存的模型权重
  - ablation_summary.json: 实验的比较摘要

## 项目特点

- **知识引导的深度学习**: 将领域知识（关联规则）与神经网络结合
- **环境特定处理**: 根据地理区域适应不同的前兆模式
- **全面评估**: 包括多个基线和详细的消融研究
- **模块化架构**: 清晰分离的组件便于扩展和实验
- **生产就绪**: 包括配置管理、日志记录、错误处理

## 调试提示

- 使用`setup_logging()`启用详细日志记录
- 检查`feasibility_check/`目录中的数据验证脚本
- 消融研究可以并行运行以加快实验速度
- 配置文件可以是YAML或JSON格式
- 所有路径相对于项目根目录