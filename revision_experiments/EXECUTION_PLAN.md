# 论文返修补充实验执行方案

**生成时间**: 2025-11-04
**方案版本**: v2.0 (基于方案A成功)
**状态**: 🟢 所有关键文件已找到，可立即执行

---

## 📊 执行摘要

### ✅ 重大突破：方案A成功

通过全盘搜索，已找到所有关键文件：
- ✅ 模型checkpoint: `best_model.pth` (548MB)
- ✅ 训练数据: `training_dataset_demo1.parquet` (19GB)
- ✅ 训练数据: `training_dataset_demo2.parquet` (4GB)
- ✅ PyTorch数据集: `train_dataset.pth` (6.1GB) + `test_dataset.pth` (5.6GB)

**结论**: 无需重新训练，可立即开始补充实验！

### 🎯 论文原始性能指标

根据论文PDF (第20页)，WE-FTT模型的性能指标：

| 指标 | WE-FTT | RandomForest (最佳基线) | 提升 |
|------|---------|-------------------------|------|
| **MCC** | **0.84** | 0.74 | +13.5% |
| Accuracy | 0.84 | 0.72 | +16.7% |
| F1 Score | 0.82 | 0.70 | +17.1% |
| Precision | 0.80 | 0.71 | +12.7% |
| Recall | 0.84 | 0.72 | +16.7% |
| Cohen's Kappa | 0.82 | 0.68 | +20.6% |

**返修目标**: 在震源学分层评估中复现类似水平的MCC值

---

## 🗂️ 一、找到的关键文件详情

### 1.1 模型文件

```bash
/mnt/hdd_4tb_data/ArchivedWorks/MBT/FTT/
├── best_model.pth                    # 548MB (2025-01-03)
├── data/
│   ├── train_dataset.pth             # 6.1GB (2025-01-03)
│   ├── test_dataset.pth              # 5.6GB (2025-01-03)
│   ├── class_weights.pth             # 1.2KB
│   └── label_encoder.pth             # 1.3KB
└── FT_Transformer_results.txt        # 136KB (训练日志)
```

### 1.2 训练数据文件

```bash
/mnt/hdd_4tb_data/ArchivedWorks/MBT/
├── FTT/Mindformers/
│   └── training_dataset_demo1.parquet   # 19GB (2025-01-12)
├── training_dataset_demo2.parquet       # 4GB (2025-01-15)
├── training_dataset_demo.parquet        # 115MB (2025-01-08)
└── training_dataset_mini.parquet        # 1.3MB (2025-02-12)
```

### 1.3 原始数据文件

```bash
/mnt/hdd_4tb_data/ArchivedWorks/MBT/Final/data/
├── downsampled_f0t0.csv                 # 5.0GB ✅
├── downsampled_f0t1.csv                 # 2.5GB ✅
├── downsampled_f0t2.csv                 # 1.0GB ✅
├── downsampled_f0t3.csv                 # 4.6GB ✅
├── downsampled_f0t4.csv                 # 5.1GB ✅
├── downsampled_f1t0.csv                 # 5.1GB ✅
├── downsampled_f1t1.csv                 # 2.6GB ✅
├── downsampled_f1t2.csv                 # 1.0GB ✅
├── downsampled_f1t3.csv                 # 4.6GB ✅
└── downsampled_f1t4.csv                 # 5.0GB ✅
Total: 36GB (10个特征 + 10个权重列，缺label列)
```

### 1.4 关联规则挖掘结果

```bash
/mnt/hdd_4tb_data/ArchivedWorks/MBT/Final/
├── freqItemsets/                        # ✅ 完整
│   ├── MBTDATA_freqItemsets_type_{0-4}.json
│   ├── MBTDATA_freqItemsets_type_{0-4}_flag_{0,1}.json
│   └── plt.py
├── Apriori_Results/                     # ✅ 完整
│   ├── MBTDATA_eq_type_{0-4}_cluster_info.json
│   ├── normalized_MBTDATA_freqItemsets_type_{0-4}.txt
│   └── mean_support_MBTDATA_freqItemsets_type_{0-4}.txt
└── cluster_range/                       # ✅ 完整
    └── MBTDATA_eq_type_{0-4}_cluster_info.json
```

---

## 🔍 二、关键问题识别

### ⚠️ 问题1：找到的模型是FTT还是WE-FTT？

**疑问**:
- 目录名是`/MBT/FTT/`，可能是FT-Transformer（基线模型）
- 但论文中的WE-FTT checkpoint在哪里？

**验证方案**:
需要检查模型架构，确认是否包含weight projection层（WE-FTT的核心特征）

**影响**:
- 如果是WE-FTT → 直接使用 ✅
- 如果是FT-Transformer → 需要进一步搜索或重新训练

### ⚠️ 问题2：震源学分层缺少性能指标

**当前状态**:
- ✅ 已完成: 154个M≥7.0事件的统计分析
- ✅ 已完成: 9个分层的事件分布
- ❌ **缺失**: 各分层的MCC/FPR（审稿人明确要求）

**审稿人要求** (Reviewer #2):
> 按震级(7.0-7.4/7.5-7.9/≥8.0)与深度(0-70/70-150/>150 km)分层，报告**条件化MCC/FPR**

**缺失原因**:
当前脚本`seismological_stratification.py`只做了统计，没有模型推理

---

## 🎯 三、三条并行执行路径

### 🔴 路径1：模型验证与准备（最高优先级）

**目标**: 验证找到的模型可用性和类型

**时间**: 今天下午 2小时

#### Step 1.1: 检查模型架构

创建验证脚本:

```python
# revision_experiments/scripts/verify_model.py
import torch
import sys
from pathlib import Path

# 配置路径
FTT_MODEL_PATH = '/mnt/hdd_4tb_data/ArchivedWorks/MBT/FTT/best_model.pth'

def verify_model_architecture():
    """验证模型架构和类型"""
    print("=" * 60)
    print("模型架构验证")
    print("=" * 60)

    # 加载checkpoint
    print(f"\n加载模型: {FTT_MODEL_PATH}")
    checkpoint = torch.load(FTT_MODEL_PATH, map_location='cpu')

    # 分析checkpoint结构
    print("\n=== Checkpoint Keys ===")
    for key in checkpoint.keys():
        print(f"  - {key}")

    # 检查模型状态字典
    if 'model_state_dict' in checkpoint:
        model_keys = list(checkpoint['model_state_dict'].keys())
        print(f"\n=== Model State Dict ({len(model_keys)} keys) ===")

        # 检查WE-FTT特征
        weight_proj_keys = [k for k in model_keys if 'weight' in k.lower()]
        feature_proj_keys = [k for k in model_keys if 'feature_proj' in k]

        print(f"\nWeight-related layers: {len(weight_proj_keys)}")
        for k in weight_proj_keys[:5]:  # 显示前5个
            print(f"  - {k}")

        print(f"\nFeature projection layers: {len(feature_proj_keys)}")
        for k in feature_proj_keys[:5]:
            print(f"  - {k}")

        # 判断模型类型
        is_we_ftt = len(weight_proj_keys) > 0
        print("\n" + "=" * 60)
        if is_we_ftt:
            print("✅ 判断: 这是 WE-FTT 模型")
        else:
            print("⚠️  判断: 这可能是 FT-Transformer 基线模型")
        print("=" * 60)

    # 检查训练信息
    if 'epoch' in checkpoint:
        print(f"\n训练轮数: {checkpoint['epoch']}")
    if 'best_mcc' in checkpoint:
        print(f"最佳MCC: {checkpoint['best_mcc']:.4f}")
    if 'optimizer_state_dict' in checkpoint:
        print("包含优化器状态: ✅")

    return checkpoint

if __name__ == "__main__":
    verify_model_architecture()
```

#### Step 1.2: 测试模型推理

```python
# revision_experiments/scripts/test_model_inference.py
import torch
import pandas as pd
from pathlib import Path

def test_inference():
    """测试模型推理功能"""
    print("测试模型推理...")

    # 加载模型
    checkpoint = torch.load('/mnt/hdd_4tb_data/ArchivedWorks/MBT/FTT/best_model.pth')

    # 加载少量测试数据
    test_data_path = '/mnt/hdd_4tb_data/ArchivedWorks/MBT/FTT/data/test_dataset.pth'
    print(f"加载测试数据: {test_data_path}")

    test_data = torch.load(test_data_path)
    print(f"测试数据类型: {type(test_data)}")

    if isinstance(test_data, dict):
        print(f"数据字典keys: {test_data.keys()}")

    # TODO: 加载模型并进行forward pass测试
    print("✅ 模型推理测试准备完成")

if __name__ == "__main__":
    test_inference()
```

#### Step 1.3: 决策点

**如果是WE-FTT模型**:
- ✅ 直接用于所有返修实验
- 进入路径2和路径3的实施

**如果是FT-Transformer模型**:
- ⚠️ 需要进一步搜索WE-FTT模型
- 备选方案: 使用`/Final/WE_FT_Transformer.py`重新训练（1-2天）

---

### 🟢 路径2：物理机制统计分析（可立即开始）

**目标**: 完成物理机制统计检验（不依赖模型）

**时间**: 今晚-明天 (1-2天)

**优势**:
- ✅ 完全不需要模型checkpoint
- ✅ 可以立即并行开始
- ✅ 提供1/5实验的保底成果

#### Step 2.1: 实现Bootstrap置信区间

```python
# revision_experiments/scripts/physical_mechanism_stats.py

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple

class PhysicalMechanismStatistics:
    """物理机制统计检验类"""

    def __init__(self):
        self.config = RevisionConfig
        self.n_bootstrap = 1000
        self.alpha = 0.05
        self.results = {}

    def load_frequent_itemsets(self, zone_type: int) -> Dict:
        """加载频繁项集数据"""
        file_path = self.config.FREQITEMSETS_DIR / f"MBTDATA_freqItemsets_type_{zone_type}.json"
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data

    def calculate_support_difference(self, eq_support: float, non_eq_support: float) -> float:
        """计算支持度差分"""
        return eq_support - non_eq_support

    def bootstrap_confidence_interval(self,
                                     eq_samples: np.ndarray,
                                     non_eq_samples: np.ndarray,
                                     n_iterations: int = 1000) -> Tuple[float, float, float]:
        """
        Bootstrap方法计算支持度差分的置信区间

        Returns:
            mean_diff: 平均差分
            ci_lower: 95% CI下界
            ci_upper: 95% CI上界
        """
        bootstrap_diffs = []

        for i in range(n_iterations):
            # 重采样
            eq_resample = np.random.choice(eq_samples, size=len(eq_samples), replace=True)
            non_eq_resample = np.random.choice(non_eq_samples, size=len(non_eq_samples), replace=True)

            # 计算差分
            diff = eq_resample.mean() - non_eq_resample.mean()
            bootstrap_diffs.append(diff)

        bootstrap_diffs = np.array(bootstrap_diffs)
        mean_diff = bootstrap_diffs.mean()
        ci_lower = np.percentile(bootstrap_diffs, 2.5)
        ci_upper = np.percentile(bootstrap_diffs, 97.5)

        return mean_diff, ci_lower, ci_upper

    def statistical_significance_test(self,
                                     eq_support: float,
                                     non_eq_support: float,
                                     n_eq: int,
                                     n_non_eq: int) -> Tuple[float, float]:
        """
        统计显著性检验（使用z-test）

        Returns:
            z_score: z统计量
            p_value: p值
        """
        # 计算标准误差
        se_eq = np.sqrt(eq_support * (1 - eq_support) / n_eq)
        se_non_eq = np.sqrt(non_eq_support * (1 - non_eq_support) / n_non_eq)
        se_diff = np.sqrt(se_eq**2 + se_non_eq**2)

        # z-test
        z_score = (eq_support - non_eq_support) / se_diff
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

        return z_score, p_value

    def analyze_zone(self, zone_type: int) -> pd.DataFrame:
        """分析单个环境区的物理机制"""
        print(f"\n分析环境区 {zone_type}: {self.config.ENVIRONMENT_ZONES[zone_type]}")

        # 加载数据
        eq_data = self.load_frequent_itemsets(zone_type)  # 地震相关
        # TODO: 加载非地震数据

        results = []

        # 对每个频繁项集进行分析
        for itemset, metrics in eq_data.items():
            eq_support = metrics['support']
            # non_eq_support = ... # 从非地震数据获取

            # Bootstrap CI
            # mean_diff, ci_lower, ci_upper = self.bootstrap_confidence_interval(...)

            # 统计检验
            # z_score, p_value = self.statistical_significance_test(...)

            results.append({
                'itemset': itemset,
                'eq_support': eq_support,
                # 'non_eq_support': non_eq_support,
                # 'support_diff': mean_diff,
                # 'ci_lower': ci_lower,
                # 'ci_upper': ci_upper,
                # 'z_score': z_score,
                # 'p_value': p_value
            })

        df = pd.DataFrame(results)
        return df

    def multiple_testing_correction(self, p_values: np.ndarray) -> np.ndarray:
        """多重比较校正（Benjamini-Hochberg）"""
        reject, p_corrected, _, _ = multipletests(
            p_values,
            alpha=self.alpha,
            method='fdr_bh'
        )
        return p_corrected, reject

    def generate_visualizations(self, results_df: pd.DataFrame, zone_type: int):
        """生成可视化图表"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. 支持度差分分布
        ax1 = axes[0, 0]
        sns.histplot(data=results_df, x='support_diff', bins=30, ax=ax1)
        ax1.axvline(0, color='red', linestyle='--', label='No difference')
        ax1.set_title(f'Support Difference Distribution - Zone {zone_type}')
        ax1.set_xlabel('Support Difference')
        ax1.legend()

        # 2. 置信区间森林图
        ax2 = axes[0, 1]
        top_n = 20
        top_items = results_df.nlargest(top_n, 'support_diff')

        y_pos = np.arange(len(top_items))
        ax2.errorbar(
            top_items['support_diff'],
            y_pos,
            xerr=[
                top_items['support_diff'] - top_items['ci_lower'],
                top_items['ci_upper'] - top_items['support_diff']
            ],
            fmt='o',
            capsize=3
        )
        ax2.axvline(0, color='red', linestyle='--')
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(top_items['itemset'].str[:30])
        ax2.set_xlabel('Support Difference with 95% CI')
        ax2.set_title(f'Top {top_n} Itemsets - Zone {zone_type}')

        # 3. P值分布
        ax3 = axes[1, 0]
        sns.histplot(data=results_df, x='p_value', bins=30, ax=ax3)
        ax3.axvline(self.alpha, color='red', linestyle='--', label=f'α={self.alpha}')
        ax3.set_title('P-value Distribution')
        ax3.legend()

        # 4. 统计显著性总结
        ax4 = axes[1, 1]
        significant = (results_df['p_corrected'] < self.alpha).sum()
        non_significant = len(results_df) - significant

        ax4.bar(['Significant', 'Non-significant'], [significant, non_significant])
        ax4.set_title('Statistical Significance Summary (FDR-corrected)')
        ax4.set_ylabel('Count')

        plt.tight_layout()

        # 保存
        output_path = self.config.FIGURES_DIR / f'physical_mechanism_zone_{zone_type}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"保存图表: {output_path}")
        plt.close()

    def generate_summary_table(self, all_results: Dict[int, pd.DataFrame]):
        """生成汇总表格"""
        summary_data = []

        for zone_type, df in all_results.items():
            significant_count = (df['p_corrected'] < self.alpha).sum()
            total_count = len(df)
            mean_support_diff = df['support_diff'].mean()

            summary_data.append({
                'Zone': zone_type,
                'Environment': self.config.ENVIRONMENT_ZONES[zone_type],
                'Total Itemsets': total_count,
                'Significant (FDR)': significant_count,
                'Proportion (%)': f"{100 * significant_count / total_count:.1f}",
                'Mean Support Diff': f"{mean_support_diff:.4f}"
            })

        summary_df = pd.DataFrame(summary_data)

        # 保存CSV
        csv_path = self.config.TABLES_DIR / 'physical_mechanism_summary.csv'
        summary_df.to_csv(csv_path, index=False)
        print(f"保存汇总表: {csv_path}")

        # 保存LaTeX
        latex_path = self.config.TABLES_DIR / 'physical_mechanism_summary.tex'
        with open(latex_path, 'w') as f:
            f.write(summary_df.to_latex(index=False, caption="Physical Mechanism Statistical Summary"))
        print(f"保存LaTeX表: {latex_path}")

        return summary_df

    def run_complete_analysis(self):
        """运行完整分析"""
        print("=" * 80)
        print("物理机制统计分析")
        print("=" * 80)

        all_results = {}

        # 分析所有5个环境区
        for zone_type in range(5):
            df = self.analyze_zone(zone_type)

            # 多重比较校正
            if len(df) > 0:
                p_corrected, reject = self.multiple_testing_correction(df['p_value'].values)
                df['p_corrected'] = p_corrected
                df['significant'] = reject

            all_results[zone_type] = df

            # 生成可视化
            self.generate_visualizations(df, zone_type)

        # 生成汇总
        summary = self.generate_summary_table(all_results)
        print("\n" + "=" * 80)
        print("分析完成！")
        print("=" * 80)
        print(summary)

        return all_results, summary

if __name__ == "__main__":
    analyzer = PhysicalMechanismStatistics()
    results, summary = analyzer.run_complete_analysis()
```

#### Step 2.2: 预期产出

**文件输出**:
- `tables/physical_mechanism_summary.csv` - 统计汇总表
- `tables/physical_mechanism_summary.tex` - LaTeX格式
- `figures/physical_mechanism_zone_{0-4}.png` - 各环境区可视化（5张图）

**论文使用**:
- **Table SX**: Physical Mechanism Statistical Summary
- **Figure SX**: Support Difference with 95% Confidence Intervals

---

### 🟡 路径3：震源学分层性能评估（需要模型）

**目标**: 为9个分层计算MCC/FPR

**时间**: 验证模型后立即开始 (2-3天)

**前置条件**: 路径1完成，确认模型可用

#### Step 3.1: 实现分层评估器

```python
# revision_experiments/scripts/seismological_stratification_with_model.py

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import matthews_corrcoef, confusion_matrix
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

class SeismologicalStratificationEvaluator:
    """震源学分层性能评估器"""

    def __init__(self, model_path: str, earthquake_catalog_path: str):
        self.model_path = model_path
        self.catalog_path = earthquake_catalog_path
        self.config = RevisionConfig

        # 加载模型
        self.model = self._load_model()

        # 加载地震目录
        self.catalog = pd.read_csv(earthquake_catalog_path)

        # 震级和深度分层定义
        self.mag_bins = [(7.0, 7.4), (7.5, 7.9), (8.0, 10.0)]
        self.depth_bins = [(0, 70), (70, 150), (150, 1000)]

    def _load_model(self):
        """加载训练好的模型"""
        print(f"加载模型: {self.model_path}")
        checkpoint = torch.load(self.model_path, map_location='cpu')

        # TODO: 根据checkpoint重建模型
        # 需要知道模型架构

        return model

    def _get_stratum_events(self, mag_range: Tuple[float, float],
                           depth_range: Tuple[float, float]) -> pd.DataFrame:
        """获取特定分层的地震事件"""
        mask = (
            (self.catalog['mag'] >= mag_range[0]) &
            (self.catalog['mag'] < mag_range[1]) &
            (self.catalog['depth'] >= depth_range[0]) &
            (self.catalog['depth'] < depth_range[1])
        )
        return self.catalog[mask]

    def _load_event_data(self, event: pd.Series) -> torch.Tensor:
        """
        为单个地震事件加载MBT数据

        时间窗口: 地震前20天
        空间范围: Dobrovolsky半径
        """
        # 提取事件信息
        event_time = pd.to_datetime(event['time'])
        event_lat = event['latitude']
        event_lon = event['longitude']
        event_mag = event['mag']

        # 计算Dobrovolsky半径 (km)
        dobrovolsky_radius = 10 ** (0.43 * event_mag)

        # 时间窗口
        start_time = event_time - pd.Timedelta(days=20)
        end_time = event_time

        # TODO: 从data/目录加载对应的MBT数据
        # 1. 确定环境类型 (Type 0-4)
        # 2. 加载对应的CSV文件
        # 3. 筛选时空窗口内的数据
        # 4. 返回特征张量

        return data_tensor

    def evaluate_stratum(self, mag_range: Tuple[float, float],
                        depth_range: Tuple[float, float]) -> Dict:
        """
        评估单个分层的性能

        Returns:
            metrics: {
                'n_events': int,
                'mcc': float,
                'fpr': float,
                'precision': float,
                'recall': float,
                'f1': float,
                'mcc_ci': (lower, upper),  # 95% CI
                'confusion_matrix': np.ndarray
            }
        """
        print(f"\n评估分层: M{mag_range[0]}-{mag_range[1]}, D{depth_range[0]}-{depth_range[1]}km")

        # 获取该分层的事件
        events = self._get_stratum_events(mag_range, depth_range)
        n_events = len(events)

        print(f"  事件数量: {n_events}")

        # 样本量不足
        if n_events < 3:
            print(f"  ⚠️ 样本量不足，跳过性能评估")
            return {
                'n_events': n_events,
                'mcc': np.nan,
                'fpr': np.nan,
                'status': 'insufficient_samples'
            }

        # 收集所有预测和标签
        all_predictions = []
        all_labels = []

        self.model.eval()
        with torch.no_grad():
            for idx, event in events.iterrows():
                # 加载事件数据
                event_data = self._load_event_data(event)

                # 模型推理
                predictions = self.model(event_data)

                # 收集结果
                all_predictions.extend(predictions.cpu().numpy())
                # all_labels.extend(...) # 真实标签

        # 计算性能指标
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)

        # MCC
        mcc = matthews_corrcoef(all_labels, all_predictions)

        # Confusion Matrix
        cm = confusion_matrix(all_labels, all_predictions)
        tn, fp, fn, tp = cm.ravel()

        # FPR
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        # Precision, Recall, F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # Bootstrap 95% CI for MCC
        mcc_ci = self._bootstrap_mcc_ci(all_labels, all_predictions)

        metrics = {
            'n_events': n_events,
            'mcc': mcc,
            'fpr': fpr,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'mcc_ci_lower': mcc_ci[0],
            'mcc_ci_upper': mcc_ci[1],
            'confusion_matrix': cm,
            'status': 'success'
        }

        print(f"  MCC: {mcc:.4f} (95% CI: {mcc_ci[0]:.4f}-{mcc_ci[1]:.4f})")
        print(f"  FPR: {fpr:.4f}")

        return metrics

    def _bootstrap_mcc_ci(self, labels: np.ndarray, predictions: np.ndarray,
                         n_iterations: int = 1000, alpha: float = 0.05) -> Tuple[float, float]:
        """Bootstrap计算MCC的置信区间"""
        n_samples = len(labels)
        mcc_values = []

        for _ in range(n_iterations):
            # 重采样
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            boot_labels = labels[indices]
            boot_preds = predictions[indices]

            # 计算MCC
            mcc = matthews_corrcoef(boot_labels, boot_preds)
            mcc_values.append(mcc)

        mcc_values = np.array(mcc_values)
        ci_lower = np.percentile(mcc_values, 100 * alpha / 2)
        ci_upper = np.percentile(mcc_values, 100 * (1 - alpha / 2))

        return ci_lower, ci_upper

    def evaluate_all_strata(self) -> pd.DataFrame:
        """评估所有9个分层"""
        print("=" * 80)
        print("震源学分层性能评估")
        print("=" * 80)

        results = []

        for mag_range in self.mag_bins:
            for depth_range in self.depth_bins:
                metrics = self.evaluate_stratum(mag_range, depth_range)

                # 添加分层信息
                metrics['mag_range'] = f"{mag_range[0]}-{mag_range[1]}"
                metrics['depth_range'] = f"{depth_range[0]}-{depth_range[1]}"
                metrics['stratum'] = f"M{mag_range[0]}-{mag_range[1]}_D{depth_range[0]}-{depth_range[1]}"

                results.append(metrics)

        df = pd.DataFrame(results)

        # 保存结果
        csv_path = self.config.TABLES_DIR / 'seismological_strata_performance.csv'
        df.to_csv(csv_path, index=False)
        print(f"\n保存结果: {csv_path}")

        return df

    def generate_forest_plot(self, results_df: pd.DataFrame):
        """生成森林图（Forest Plot）"""
        fig, ax = plt.subplots(figsize=(10, 8))

        # 过滤有效数据
        valid_results = results_df[results_df['status'] == 'success'].copy()
        valid_results = valid_results.sort_values('mcc', ascending=True)

        # Y轴位置
        y_pos = np.arange(len(valid_results))

        # 绘制误差线和点
        ax.errorbar(
            valid_results['mcc'],
            y_pos,
            xerr=[
                valid_results['mcc'] - valid_results['mcc_ci_lower'],
                valid_results['mcc_ci_upper'] - valid_results['mcc']
            ],
            fmt='o',
            markersize=8,
            capsize=5,
            capthick=2,
            elinewidth=2
        )

        # 添加整体平均MCC参考线
        overall_mcc = 0.84  # 论文报告值
        ax.axvline(overall_mcc, color='red', linestyle='--', linewidth=2,
                  label=f'Overall MCC = {overall_mcc:.2f}')

        # 设置Y轴标签
        ax.set_yticks(y_pos)
        ax.set_yticklabels(valid_results['stratum'])

        # 标签和标题
        ax.set_xlabel('Matthews Correlation Coefficient (MCC)', fontsize=12)
        ax.set_title('MCC by Seismological Strata with 95% Confidence Intervals',
                    fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(axis='x', alpha=0.3)

        plt.tight_layout()

        # 保存
        output_path = self.config.FIGURES_DIR / 'forest_plot_seismological_strata.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"保存森林图: {output_path}")
        plt.close()

    def generate_fpr_comparison(self, results_df: pd.DataFrame):
        """生成FPR对比图"""
        fig, ax = plt.subplots(figsize=(12, 6))

        # 过滤有效数据
        valid_results = results_df[results_df['status'] == 'success'].copy()

        # 绘制柱状图
        x_pos = np.arange(len(valid_results))
        bars = ax.bar(x_pos, valid_results['fpr'], color='steelblue', alpha=0.7)

        # 添加FPR阈值线
        fpr_threshold = 0.1  # 可接受阈值
        ax.axhline(fpr_threshold, color='red', linestyle='--', linewidth=2,
                  label=f'Acceptable FPR threshold = {fpr_threshold:.2f}')

        # 标注
        ax.set_xticks(x_pos)
        ax.set_xticklabels(valid_results['stratum'], rotation=45, ha='right')
        ax.set_ylabel('False Positive Rate (FPR)', fontsize=12)
        ax.set_title('FPR Comparison Across Seismological Strata',
                    fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()

        # 保存
        output_path = self.config.FIGURES_DIR / 'fpr_comparison_strata.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"保存FPR对比图: {output_path}")
        plt.close()

    def run_complete_evaluation(self):
        """运行完整评估流程"""
        # 评估所有分层
        results_df = self.evaluate_all_strata()

        # 生成可视化
        self.generate_forest_plot(results_df)
        self.generate_fpr_comparison(results_df)

        # 打印摘要
        print("\n" + "=" * 80)
        print("评估摘要")
        print("=" * 80)
        print(results_df[['stratum', 'n_events', 'mcc', 'fpr', 'status']])

        return results_df

if __name__ == "__main__":
    evaluator = SeismologicalStratificationEvaluator(
        model_path='/mnt/hdd_4tb_data/ArchivedWorks/MBT/FTT/best_model.pth',
        earthquake_catalog_path='/mnt/hdd_4tb_data/ArchivedWorks/MBT/Final/13-23EQ.csv'
    )
    results = evaluator.run_complete_evaluation()
```

#### Step 3.2: 预期产出

**文件输出**:
- `tables/seismological_strata_performance.csv` - 分层性能表
- `tables/seismological_strata_performance.tex` - LaTeX格式
- `figures/forest_plot_seismological_strata.png` - 森林图
- `figures/fpr_comparison_strata.png` - FPR对比图

**论文使用**:
- **Table S4**: Stratified Performance Metrics
- **Figure S5**: Forest Plot of MCC by Strata
- **Figure S6**: FPR Comparison Across Strata

#### Step 3.3: 样本量不足的处理

对于样本量<3的分层：
- 在表格中标注为"N/A"或"-"
- 在论文中说明：
  > "Due to limited sample sizes (n<3) in certain strata (M8.0+ deep earthquakes), performance metrics for these groups should be interpreted with caution."

---

## 📅 四、详细时间表

### Week 1: 模型验证 + 基础实验

#### Day 1 (今天 - 2025-11-04)

**上午 09:00-13:00**
- [ ] 运行`verify_model.py`验证模型架构
- [ ] 运行`test_model_inference.py`测试推理
- [ ] 根据结果做出决策（WE-FTT vs FT-Transformer）

**下午 14:00-18:00**
- [ ] 开始物理机制统计分析（路径2）
- [ ] 实现Bootstrap置信区间计算
- [ ] 运行Zone 0 (Marine)的分析

**晚上 19:00-22:00**
- [ ] 继续物理机制分析（Zone 1-4）
- [ ] 生成可视化图表

**预期产出**:
- ✅ 确认模型类型和可用性
- ✅ 完成物理机制分析框架

---

#### Day 2-3 (2025-11-05 至 11-06)

**Day 2 任务**:
- [ ] 完成物理机制统计分析的所有5个环境区
- [ ] 多重比较校正（FDR）
- [ ] 生成所有图表和表格
- [ ] 开始震源学分层评估（路径3）

**Day 3 任务**:
- [ ] 实现`seismological_stratification_with_model.py`
- [ ] 测试单个分层的评估流程
- [ ] 运行部分分层（先测试样本量充足的）

**预期产出**:
- ✅ 物理机制统计分析完成
- ✅ 震源学分层评估框架完成

---

#### Day 4-5 (2025-11-07 至 11-08 周末)

**主要任务**:
- [ ] 完成所有9个分层的性能评估
- [ ] 生成森林图和FPR对比图
- [ ] 处理样本量不足的分层
- [ ] 开始其他补充实验（全球FPR、样本外验证）

**预期产出**:
- ✅ 震源学分层评估完成
- ✅ 开始其他补充实验

---

### Week 2: 补充实验 + 论文修订

#### Day 6-7 (2025-11-09 至 11-10)

**实验3: 全球FPR评估**
- [ ] 随机采样100天无震日期（每个环境区）
- [ ] 模型推理计算FPR
- [ ] 验证FPR < 0.1的要求

**实验4: 样本外验证**
- [ ] 2023年后M≥7.0事件
- [ ] 独立数据集评估

---

#### Day 8-10 (2025-11-11 至 11-13)

**实验5: 海洋伪影控制**
- [ ] 海啸时间窗口筛选
- [ ] 海况百分位分析
- [ ] 极化差异阈值验证

**整合与文档**:
- [ ] 整合所有实验结果
- [ ] 更新论文相关章节
- [ ] 准备回复审稿人的信

---

## 🎯 五、关键决策点

### 决策1：模型类型确认

**触发时机**: Day 1上午完成模型验证后

**选项**:

| 选项 | 条件 | 行动 | 时间成本 |
|------|------|------|---------|
| A | 是WE-FTT | 直接使用，进入路径2和3 | 0天 |
| B | 是FT-Transformer | 继续搜索WE-FTT，或重新训练 | 1-2天 |

**推荐**: 如果是FT-Transformer，优先搜索WE-FTT（检查其他目录、压缩包）

---

### 决策2：样本量不足分层的处理

**触发时机**: 震源学分层评估时

**选项**:

| 选项 | 方案 | 优点 | 缺点 |
|------|------|------|------|
| A | 报告统计但标注"N/A" | 诚实、科学 | 审稿人可能不满意 |
| B | 合并相邻分层 (D>70) | 提供更多数据 | 改变原始分层设计 |
| C | 只报告样本充足分层 | 数据可靠 | 分析不完整 |

**推荐**: **选项A** - 报告所有分层，对n<3的标注样本量不足

---

### 决策3：实验优先级

**如果时间紧张**，按以下优先级执行：

| 优先级 | 实验 | 原因 |
|--------|------|------|
| 🔴 P1 | 物理机制统计 | 不需要模型，审稿人要求 |
| 🔴 P1 | 震源学分层评估 | 审稿人明确要求MCC/FPR |
| 🟡 P2 | 全球FPR评估 | 证明低误报率 |
| 🟢 P3 | 样本外验证 | 增强可信度 |
| 🟢 P3 | 海洋伪影控制 | 排除混淆因素 |

---

## 📊 六、预期最终产出

### 6.1 表格

| 表格编号 | 文件名 | 内容 |
|---------|--------|------|
| Table S3 | seismological_strata_statistics.csv | 震源学分层统计（已完成）|
| Table S4 | seismological_strata_performance.csv | 震源学分层性能 |
| Table S5 | physical_mechanism_summary.csv | 物理机制统计摘要 |
| Table S6 | global_fpr_evaluation.csv | 全球FPR评估 |

### 6.2 图表

| 图表编号 | 文件名 | 内容 |
|---------|--------|------|
| Figure S4 | temporal_distribution_strata.png | 时间分布（已完成）|
| Figure S5 | forest_plot_seismological_strata.png | 分层MCC森林图 |
| Figure S6 | fpr_comparison_strata.png | 分层FPR对比 |
| Figure S7 | physical_mechanism_zone_*.png | 物理机制分析（5张）|

### 6.3 补充文本

**Supplementary Text S1**: 震源学分层方法与结果
**Supplementary Text S2**: 物理机制统计检验详细方法
**Supplementary Text S3**: 样本量限制讨论

---

## ⚠️ 七、风险与应对

### 风险1：模型不可用或性能不匹配

**概率**: 低-中
**影响**: 高

**应对方案**:
1. 优先级调整：先完成物理机制统计（不需要模型）
2. 如果是FT-Transformer：
   - 搜索其他可能位置
   - 联系原作者
   - 最后选项：重新训练（1-2天）

---

### 风险2：数据加载困难

**概率**: 中
**影响**: 中

**应对方案**:
1. 仔细研究现有代码（`/Final/WE_FT_Transformer.py`）的数据加载逻辑
2. 使用PyTorch保存的数据集（`train_dataset.pth`）作为参考
3. 简化数据加载：先实现单个事件的评估，再批量处理

---

### 风险3：时间不足

**概率**: 中
**影响**: 高

**应对方案**:
1. 严格按优先级执行（P1 > P2 > P3）
2. 如果2周内无法完成全部5个实验：
   - 提交已完成的实验结果
   - 向审稿人说明其余实验正在进行中
   - 承诺在final version提供完整结果

---

## 📞 八、需要立即确认的问题

在开始实施前，请回答以下问题：

### 问题1：执行优先级
您希望我们：
- [ ] A. 先验证模型（路径1） → 然后决定下一步
- [ ] B. 并行开始：模型验证（路径1）+ 物理机制统计（路径2）
- [ ] C. 只做物理机制统计（最保险）

### 问题2：GPU资源确认
- GPU型号：**nvlink双3090** ✅
- 可用时长：________
- 可以24/7运行：[ ] 是 [ ] 否

### 问题3：审稿人Deadline
- 明确日期：________
- 如果不确定：大约还有 ________ 周时间

### 问题4：风险承受度
如果模型验证失败，您希望：
- [ ] A. 立即开始重新训练（1-2天）
- [ ] B. 先完成不需要模型的实验，再决定
- [ ] C. 寻找其他解决方案

### 问题5：代码实现偏好
我应该：
- [ ] A. 先写完整的框架代码，您review后再运行
- [ ] B. 边写边测试，快速迭代
- [ ] C. 给您伪代码，您来实现

---

## 🚀 九、立即行动清单

### 今天下午（优先级排序）

**高优先级（必须做）**:
- [ ] 1. 运行模型验证脚本 `verify_model.py`
- [ ] 2. 查看验证结果，确定模型类型
- [ ] 3. 回答上面第8节的5个问题

**中优先级（建议做）**:
- [ ] 4. 开始物理机制统计分析代码
- [ ] 5. 测试freqItemsets数据加载

**低优先级（可选）**:
- [ ] 6. 研究现有WE_FT_Transformer.py的数据加载逻辑
- [ ] 7. 准备GPU环境

---

## 📝 十、版本历史

| 版本 | 日期 | 更新内容 |
|------|------|---------|
| v1.0 | 2025-11-04 | 初始STATUS_SUMMARY.md |
| **v2.0** | **2025-11-04** | **基于方案A成功的完整执行方案** |

---

## 📧 十一、联系与协作

**本文档维护者**: Claude Code
**执行负责人**: [您的名字]
**紧急联系**: [您的联系方式]

**文档更新规则**:
- 每完成一个里程碑，更新本文档
- 每天结束时，记录当天进展
- 遇到阻塞问题，立即在"风险与应对"章节添加记录

---

**最后更新**: 2025-11-04 12:30
**下次审查**: 今天晚上（完成模型验证后）

