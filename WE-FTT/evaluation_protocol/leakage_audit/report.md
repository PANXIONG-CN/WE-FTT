# T1 权重列/聚类/关联规则泄漏审计报告（v1）

生成日期：2026-02-23

## 结论（必须回答的二元结论）

- 权重列 `*_cluster_labels_weight` 是否在全量数据上预计算并写入？
  - **是（已确认“以静态列形式写入 parquet/CSV”）**：`training_dataset.parquet` 与 `downsampled_f*.csv` 均包含这些列；训练脚本不会在 split 之后重新计算。若这些权重/离散化映射在生成 parquet 时使用了全量样本（含测试集），则构成高概率泄漏（尤其是权重生成使用了标签信息）。
- KMeans/Apriori 是否在包含测试集的数据上拟合/挖掘？
  - **在当前“结果复现路径”中无法从训练脚本侧保证否**：训练脚本（例如 `scripts/ft_transformer_best.py`）直接消费带权重列的 parquet，并在其内部做随机 split，不包含权重生成步骤；因此 **权重生成发生在 split 之前**。如果该步骤使用了全量数据（极常见），则属于明确的 train/test 信息泄漏。

> 说明：仓库内 `scripts/run_preprocessing.py` 的实现是“先 split，再在 train 上做知识挖掘并应用到 val/test”，逻辑上是正确方向；但当前用于复现的外部数据产物/训练脚本路径并未绑定该流程，导致审计结论必须以“数据产物 + 训练入口的现实执行链”为准。

## 证据链（可复核）

### 1) 数据产物层面：权重列作为静态输入存在

- 外部训练数据：`/mnt/hdd_4tb_data/ArchivedWorks/MBT/FTT/updated_code/training_dataset.parquet`
  - 列包含：10 通道 `BT_*`、10 个 `*_cluster_labels_weight`、以及 `label`。
  - 权重列取值为离散集合（例如在首个 row_group 中每列仅出现少量 unique 值），符合“先离散化→按簇/规则映射写入”的特征。
  - 快照工具：`WE-FTT/evaluation_protocol/leakage_audit/inspect_weights.py`
- 仓库内背景数据：`WE-FTT/data/processed/downsampled_f*.csv`
  - 也包含 `*_cluster_labels_weight` 列（示例行中为 1.0），说明权重列在“进入训练/采样”阶段之前已存在于数据文件中。

### 2) 训练入口层面：split 发生在消费 parquet 之后

- `WE-FTT/scripts/ft_transformer_best.py`
  - `ParquetDataset` 直接读取 parquet 中的 `features` 与 `weights` 列。
  - `load_data()` 内使用随机索引划分 train/test（非事件级，且在权重生成之外）。
  - 训练脚本未包含任何“基于 train 拟合权重映射并应用到 test”的逻辑。

### 3) 算法实现层面：关联挖掘显式使用标签信息

- `WE-FTT/src/association_mining.py`
  - `KnowledgeMiner._prepare_transactions()` 将 `label_{row[label_column]}` 追加到每条 transaction。
  - 这意味着权重生成是 **监督式** 的：如果在包含测试集的样本上挖掘/计算支持度差，会把测试集标签信息泄漏进特征权重。

### 4) 数据切分实现层面：默认是样本级 split（不是事件级）

- `WE-FTT/src/data_processing.py`
  - `DataSplitter.split_data()` 使用 `train_test_split(... stratify=y)` 做样本级划分。
  - 同一地震事件的多个时空样本可能跨集合 → 即使权重按 train-only 生成，也会存在事件泄漏（T2）。

## 风险评估

- **标签泄漏（致命）**：权重生成显式使用标签信息；若权重在全量数据上生成并写入 parquet，则测试集标签信息可通过权重列泄漏到训练过程。
- **事件泄漏（严重）**：样本级随机 split 无法阻止同一事件的相关样本跨集合，导致过高的泛化估计。

## 修复方案（DoD 对齐｜推荐落地路径）

### 修复原则（必须满足）

1. **先切分（事件级）**：同一 `event_id` 的样本不得跨 train/val/test。
2. **训练折内拟合**：离散化（KMeans）与权重映射仅在 train 折拟合。
3. **应用到 val/test**：使用 train 折产出的映射，对 val/test 生成权重列，不可重拟合。
4. **产物可审计**：映射（簇中心、簇权重表、随机种子）需要落盘。

### 最小可用实现（已提供脚本）

- 事件级切分清单：`WE-FTT/evaluation_protocol/data_splits/make_event_splits.py`
- 训练折内权重生成（v1，KMeans + support_diff）：`WE-FTT/evaluation_protocol/leakage_audit/foldwise_weighting.py`

该实现满足 DoD 的数据流顺序，并把“泄漏风险”变成可复核的映射工件（`artifacts.meta.json`）。

## 下一步（与 T2/T4/T7 的衔接）

- 使用 `event_grouped_splits_v1.json` 作为全套实验（Placebo/残差化/ERA5）的唯一切分来源，保证不同实验之间可横向比较且可审计。
- Placebo/残差化/ERA5 的任何拟合步骤（回归、标准化、阈值选择）都必须以 train 折为“唯一信息源”，并对 val/test 仅做 transform/apply。

