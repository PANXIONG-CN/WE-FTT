# Table S2 Caption
# Table S2 题注：样本外事件性能指标

---

## Table S2 | Out-of-Sample Event Performance Metrics

### English

**Comprehensive performance metrics for forward-looking validation on 26 real post-training M≥7.0 earthquakes from USGS catalog (August 2023 to September 2025)**

This table presents detailed classification performance for each earthquake event, demonstrating the model's temporal robustness under true out-of-sample conditions with frozen weights (no retraining). The table is organized into the following columns:

**Event Identification**:
- **Event ID**: USGS earthquake identifier (abbreviated, first 15 characters followed by ellipsis)
- **Date**: Earthquake occurrence date (YYYY-MM-DD format)
- **Location**: Geographic description from USGS catalog
- **Magnitude**: Earthquake magnitude in Richter scale (M7.0–8.8 range)
- **Depth (km)**: Focal depth in kilometers (6–639.5 km range)
- **Zone**: Environmental zone classification (A: Marine, B: Humid Forest, C: Dry Forest, D: Wetland, E: Arid Land)

**Performance Metrics**:
- **MCC**: Matthews Correlation Coefficient (range: 0.779–0.820, mean: 0.802 ± 0.013)
- **F1**: F1 score (range: 0.780–0.808, mean: 0.788 ± 0.009)
- **Precision**: Positive predictive value (range: 0.760–0.786, mean: 0.767 ± 0.008)
- **Recall**: True positive rate/sensitivity (range: 0.800–0.825, mean: 0.808 ± 0.008)

### Statistical Summary

**Temporal Coverage**:
- Total Events: 26
- Date Range: 2023-08-29 to 2025-09-19 (25 months)
- Temporal Distribution: 1 event (Aug 2023), 2 events (Nov-Dec 2023), 7 events (2024), 16 events (2025)

**Performance Comparison with In-Sample Baseline**:

| Metric | In-Sample (Training) | Out-of-Sample (Validation) | Difference (Δ) | Relative Change |
|--------|---------------------|---------------------------|----------------|-----------------|
| MCC | 0.840 | 0.802 | -0.038 | -4.5% |
| F1 | 0.820 | 0.788 | -0.032 | -3.9% |
| Precision | 0.800 | 0.767 | -0.033 | -4.1% |
| Recall | 0.840 | 0.808 | -0.032 | -3.8% |

**Key Observations**:
1. **Consistent Performance Degradation**: All metrics show 3.8-4.5% degradation, indicating uniform temporal robustness without selective metric collapse
2. **Maintained Recall Priority**: Recall (0.808) remains higher than Precision (0.767), preserving the model's emphasis on sensitivity crucial for earthquake early warning
3. **Low Standard Deviations**: MCC SD = 0.013, F1 SD = 0.009, indicating stable performance across diverse events
4. **No Overfitting Evidence**: Modest degradation (<5%) suggests good generalization without training set memorization

**Geographic and Tectonic Diversity**:
- **Subduction Zones** (15 events, 58%): Pacific Ring of Fire (Kamchatka, Alaska, Japan, Philippines, Vanuatu, Peru, Chile)
- **Collision Boundaries** (2 events, 8%): Myanmar, Tibet
- **Transform/Extensional** (9 events, 34%): Drake Passage, mid-ocean ridges, intraplate

**Environmental Zone Distribution**:
- Zone A (Marine): 20 events (77%), mean MCC = 0.800
- Zone B (Humid Forest): 1 event (4%), MCC = 0.820
- Zone D (Wetland): 2 events (8%), mean MCC = 0.806
- Zone E (Arid): 3 events (11%), mean MCC = 0.787

**Depth Distribution**:
- Shallow (0-70 km): 15 events, mean MCC = 0.803
- Intermediate (70-150 km): 8 events, mean MCC = 0.799
- Deep (>150 km): 3 events, mean MCC = 0.803

**Magnitude Distribution**:
- M7.0-7.4: 18 events, mean MCC = 0.800
- M7.5-7.9: 7 events, mean MCC = 0.806
- M≥8.0: 1 event (M8.8), MCC = 0.794

### Validation Protocol

**Data Source**: Events obtained from USGS Earthquake Catalog (https://earthquake.usgs.gov/) via API query on 2025-11-12. Query parameters: starttime=2023-08-01, endtime=2025-09-30, minmagnitude=7.0, format=geojson.

**Model Configuration**: Weights frozen from training phase (no fine-tuning or retraining). Identical evaluation protocol as main text: same feature extraction, preprocessing, and classification thresholds.

**Performance Calculation**: Metrics computed using identical methodology as in-sample evaluation to ensure fair comparison. Each event treated as independent test case with 1000 samples per event for statistical robustness.

**Validation Type**: True forward-looking (prospective) validation - all events occurred after model training completion, ensuring no temporal data leakage.

### Interpretation

The minimal performance degradation (4-5%) across all metrics validates the model's temporal robustness and generalization capability. The consistent degradation pattern (as opposed to erratic fluctuations) suggests the model has learned physically-grounded decision boundaries rather than dataset-specific artifacts. The preservation of high recall (0.808) is particularly important for operational earthquake early warning systems, where false negatives (missed earthquakes) are more costly than false positives (false alarms). The slight decrease in precision (0.767) represents an acceptable trade-off for maintaining sensitivity in out-of-sample scenarios.

The stable performance across diverse tectonic settings (subduction, collision, transform), environmental zones (marine-dominant but including terrestrial), and depth ranges (shallow to very deep) demonstrates that the environment-specific knowledge integration approach successfully captured transferable precursor patterns rather than overfitting to training period characteristics.

---

### 中文

**基于USGS目录26个训练后M≥7.0地震（2023年8月至2025年9月）的前瞻性验证综合性能指标**

该表格呈现了每个地震事件的详细分类性能，展示了模型在冻结权重（无重训练）的真实样本外条件下的时间稳健性。表格组织为以下列：

**事件识别**：
- **Event ID**：USGS地震标识符（缩写，前15个字符后跟省略号）
- **Date**：地震发生日期（YYYY-MM-DD格式）
- **Location**：来自USGS目录的地理描述
- **Magnitude**：里氏震级（M7.0–8.8范围）
- **Depth (km)**：震源深度（千米，6–639.5公里范围）
- **Zone**：环境区域分类（A：海洋，B：湿润森林，C：干燥森林，D：湿地，E：干旱地）

**性能指标**：
- **MCC**：Matthews相关系数（范围：0.779–0.820，平均值：0.802 ± 0.013）
- **F1**：F1分数（范围：0.780–0.808，平均值：0.788 ± 0.009）
- **Precision**：精度/阳性预测值（范围：0.760–0.786，平均值：0.767 ± 0.008）
- **Recall**：召回率/真阳性率/灵敏度（范围：0.800–0.825，平均值：0.808 ± 0.008）

### 统计摘要

**时间覆盖**：
- 总事件数：26
- 日期范围：2023-08-29至2025-09-19（25个月）
- 时间分布：1个事件（2023年8月），2个事件（2023年11-12月），7个事件（2024年），16个事件（2025年）

**与样本内基线的性能比较**：

| 指标 | 样本内（训练） | 样本外（验证） | 差异 (Δ) | 相对变化 |
|------|---------------|---------------|---------|---------|
| MCC | 0.840 | 0.802 | -0.038 | -4.5% |
| F1 | 0.820 | 0.788 | -0.032 | -3.9% |
| 精度 | 0.800 | 0.767 | -0.033 | -4.1% |
| 召回率 | 0.840 | 0.808 | -0.032 | -3.8% |

**关键观察**：
1. **一致的性能退化**：所有指标显示3.8-4.5%的退化，表明统一的时间稳健性，没有选择性指标崩溃
2. **保持召回率优先**：召回率（0.808）保持高于精度（0.767），保留了模型对地震早期预警至关重要的灵敏度的重视
3. **低标准偏差**：MCC SD = 0.013，F1 SD = 0.009，表明跨不同事件的稳定性能
4. **无过拟合证据**：适度退化（<5%）表明良好的泛化能力，没有训练集记忆

**地理和构造多样性**：
- **俯冲带**（15个事件，58%）：环太平洋火山带（堪察加、阿拉斯加、日本、菲律宾、瓦努阿图、秘鲁、智利）
- **碰撞边界**（2个事件，8%）：缅甸、西藏
- **转换/伸展**（9个事件，34%）：德雷克海峡、洋中脊、板内

**环境区域分布**：
- A区（海洋）：20个事件（77%），平均MCC = 0.800
- B区（湿润森林）：1个事件（4%），MCC = 0.820
- D区（湿地）：2个事件（8%），平均MCC = 0.806
- E区（干旱）：3个事件（11%），平均MCC = 0.787

**深度分布**：
- 浅源（0-70公里）：15个事件，平均MCC = 0.803
- 中等深度（70-150公里）：8个事件，平均MCC = 0.799
- 深源（>150公里）：3个事件，平均MCC = 0.803

**震级分布**：
- M7.0-7.4：18个事件，平均MCC = 0.800
- M7.5-7.9：7个事件，平均MCC = 0.806
- M≥8.0：1个事件（M8.8），MCC = 0.794

### 验证协议

**数据来源**：事件从USGS地震目录（https://earthquake.usgs.gov/）通过API查询获得，查询日期2025-11-12。查询参数：starttime=2023-08-01，endtime=2025-09-30，minmagnitude=7.0，format=geojson。

**模型配置**：权重从训练阶段冻结（无微调或重训练）。与主文相同的评估协议：相同的特征提取、预处理和分类阈值。

**性能计算**：使用与样本内评估相同的方法计算指标以确保公平比较。每个事件作为独立测试案例，每个事件1000个样本以确保统计稳健性。

**验证类型**：真实前瞻性（前瞻）验证——所有事件发生在模型训练完成之后，确保没有时间数据泄漏。

### 解释

所有指标的最小性能退化（4-5%）验证了模型的时间稳健性和泛化能力。一致的退化模式（而非不规则波动）表明模型学习了基于物理的决策边界，而非数据集特定的伪影。高召回率的保持（0.808）对于操作性地震早期预警系统特别重要，其中假阴性（遗漏地震）比假阳性（误报）代价更高。精度的轻微下降（0.767）代表了在样本外场景中维持灵敏度的可接受权衡。

跨不同构造环境（俯冲、碰撞、转换）、环境区域（海洋为主但包括陆地）和深度范围（浅源到深源）的稳定性能表明，环境特定知识集成方法成功捕获了可迁移的前兆模式，而不是过度拟合训练期特征。

---

## Notes

- All performance metrics calculated using identical evaluation protocol as main text to ensure comparability
- Model weights frozen from training phase - no retraining, fine-tuning, or parameter adjustment
- Events represent true out-of-sample data - all occurred after model training completion (post-August 2023)
- Environmental zone assignments based on geographic location using simplified classification rules
- Dobrovolsky radius calculated as R = 10^(0.43M) km for each event

---

**Table Generation**: Programmatically generated using Python 3.9+ with pandas. Source code: `exp2/scripts/generate_table_s2.py`.

**Data Validation**: All event information cross-verified with USGS authoritative catalog. Performance metrics validated against in-sample baseline using identical computation methods.
