# WE-FTT 论文 AI 审稿意见综合分析与 TGRS 转投执行方案

> 生成日期：2026-02-23
> 基于材料：`main_clean.tex`（第二轮修改稿）、`response_to_reviewers.tex`（第二轮回复信）、`gemini.md`（Gemini AI 审稿）、`gpt.md`（GPT AI 审稿）、项目代码库交叉核实

---

## 一、背景与决策基础

### 1.1 论文状态
- **期刊**：Remote Sensing of Environment (RSE)，两轮审稿后被拒
- **拒稿核心原因**：AE 认定 "Tb effects (0.4–2.8K) indistinguishable from instrument noise"，且"fails to establish signals are genuinely seismic"
- **第二轮已做改动**：Fresnel 灵敏度对比（Table S5）、P-hole 措辞弱化、浅表感知限制声明、定量介电对比

### 1.2 用户决策
| 决策点 | 选择 |
|--------|------|
| 目标期刊 | **IEEE TGRS**（遥感顶刊，需接近理想补强包） |
| Table S5 | **重新包装**（化"自证不可测"为方法学动机） |
| 海洋区 | **补 ERA5 协变量分析**（条件化后证明 FPR 下降） |
| 预震/后震拆分 | **暂缓**（保持当前 -20~+10d 标注） |
| 时间规划 | **1-2 月深度补强** |

---

## 二、AI 审稿意见分类评估

### 2.1 可取的建议（经代码核实为真实缺口，全盘接受）

| # | 建议内容 | 来源 | 严重程度 | 代码核实结果 |
|---|----------|------|----------|------------|
| 1 | 不申诉，转投 | Gemini+GPT | — | 共识100%，AE终局措辞不可推翻 |
| 2 | 信号/噪声叙事重构 | Gemini+GPT | 致命 | ✅ 模型实际是高维特征空间模式识别，但论文叙事导致审稿人误读为单像素阈值检测 |
| 3 | Placebo/负对照实验 | Gemini+GPT | 致命 | ✅ 代码中**零** placebo 实现；`feasibility_check/` 仅有雏形 |
| 4 | 物理主张降维 | Gemini+GPT | 可改写 | ✅ 第二轮已改但不够彻底，仍有"precursor/early warning"措辞 |
| 5 | 海洋区协变量分析 | Gemini+GPT | 严重 | ✅ 确认无 ERA5 协变量代码 |
| 6 | 事件级/区域级严格切分 | Gemini+GPT | 严重 | ✅ 当前80/20 split未明确按事件分组 |
| 7 | 噪声预算+时空聚合降噪论证 | Gemini+GPT | 致命 | ✅ 无相关计算代码 |
| 8 | 环境混杂残差化 | GPT(MVS-2) | 严重 | ✅ 无协变量回归/残差化实现 |
| 9 | 前瞻验证提升为主结果 | Gemini+GPT | 改写 | ✅ 已有26事件MCC≈0.802但仅在SI |
| 10 | 公开评估协议 | GPT(GS-11) | 加分 | ✅ 代码已开源但缺评估协议脚本 |
| 11 | 半物理辐射传输敏感性 | GPT(GS-6) | TGRS需要 | ✅ 无τ-ω模型实现 |

### 2.2 过度工程化的建议（降低优先级或放入"未来工作"）

| # | 建议内容 | 来源 | 不采纳理由 |
|---|----------|------|-----------|
| 1 | 跨传感器多参数共证(GNSS TEC/MODIS LST) | GPT(GS-7) | 工作量巨大，非拒稿核心原因。写入"未来工作" |
| 2 | 多尺度分辨率实验(0.1°/0.25°/0.5°) | GPT(GS-8) | AMSR-2足迹固定，做更高分辨率需全新数据链。Limitations中声明 |
| 3 | DiD/合成对照框架 | GPT(GS-9) | 遥感+地震领域非标准要求，Placebo已足够 |
| 4 | 板块级空间隔离(Leave-One-Tectonic-Plate-Out) | Gemini(项7) | 154个M≥7.0事件按板块切分后统计功效不足。改为10°×10°块 |
| 5 | 7天极速转投 | Gemini | 对TGRS不现实。1-2月更务实 |

### 2.3 AI 审稿遗漏的关键风险（需额外补充）

| # | 风险点 | 发现来源 | 严重程度 | 说明 |
|---|--------|----------|----------|------|
| 1 | **权重列数据泄漏** | 代码审计 | ⚠️⚠️⚠️ 致命 | 消融实验显示移除权重投影后MCC从~0.75暴跌至~0.32。权重来自地震/非地震数据的支持度差——若在全部数据上计算则存在严重标签泄漏。**两份AI审稿均未提及** |
| 2 | 两份AI审稿低估前瞻验证价值 | 分析判断 | 叙事遗漏 | 26事件前瞻验证(MCC≈0.802)是最强因果证据，应提升为主结果 |

---

## 三、面向 TGRS 的分阶段执行方案（8 周）

### 第 1 阶段：基础审计与数据准备（第 1-2 周）

#### T1. 权重生成流程泄漏审计 ⚠️ P0 阻断级

**背景**：WE-FTT 的核心创新是将关联规则挖掘得到的特征权重集成到 Transformer 中。权重列 `{feature}_cluster_labels_weight` 通过以下流程生成：
1. K-means 聚类将连续 BT 特征离散化为簇 ID
2. Apriori 算法分别在地震样本和非地震样本上挖掘频繁项集
3. 计算支持度差（地震 vs 非地震），作为特征权重

**泄漏风险点**：
- 若 K-means 聚类在全部数据（含测试集）上训练 → 测试集特征分布信息泄漏
- 若 Apriori 挖掘使用全部数据 → 测试集标签信息直接泄漏
- 若权重是预计算好存在 parquet 文件中的静态列 → 几乎确定存在泄漏

**审计步骤**：
1. 检查 `src/association_mining.py` 中 `ClusterAnalyzer` 和 `AprioriMiner` 的调用时机
2. 检查 `scripts/train.py` 中数据切分与权重生成的执行顺序
3. 检查 `data/processed/downsampled_f*.parquet` 是否已包含预计算的权重列
4. 追踪 `src/config.py` 中 `COLUMNS_WEIGHTS` 的使用路径

**若发现泄漏（高概率）**：
- 修改 pipeline：数据切分 → 仅在训练集上做 K-means + Apriori → 生成权重 → 应用到验证/测试集
- 重跑全部实验（主实验 + 消融 + 基线对比）
- 更新所有结果表格和图表

**关键文件**：
- `src/association_mining.py` (ClusterAnalyzer, AprioriMiner, WeightCalculator, KnowledgeMiner)
- `scripts/train.py`
- `src/config.py` (COLUMNS_WEIGHTS, DataProcessingConfig)

**输出**：泄漏审计报告 + 修复后的 pipeline 代码 + 重跑结果

---

#### T2. 事件级严格数据切分（对应 MVS-4）

**目标**：实现事件感知的数据切分，防止同一地震事件的样本泄漏到不同数据集

**设计**：
```
1. 为每条样本标注所属地震事件 ID
2. 按事件 ID 分组（同一事件的所有时空样本归为一组）
3. Dobrovolsky 半径缓冲区内的样本必须归属同一组
4. 80/10/10 切分（按事件组，非按样本）
5. 关联规则挖掘在训练折内独立执行

区域外推测试：
- 将全球划分为 10°×10° 空间块
- 轮流留出一个块作为纯盲测区域
- 报告每个区域的性能与失败案例
```

**修改文件**：
- `src/config.py`：增加事件级切分配置
- `scripts/train.py`：实现 `EventGroupedSplitter` 类
- 新增 `scripts/run_spatial_cv.py`：区域留一交叉验证

**输出**：事件级切分方案 + 代码 + 区域留一评估结果

---

### 第 2 阶段：核心补强实验（第 3-5 周）

#### T4. Placebo/安慰剂实验（对应 MVS-3）⚠️ 核心

**目标**：严格证明模型学到的是地震相关信号而非环境混杂/伪相关

**4 类 Placebo 设计**：

| 类型 | 设计 | 对照原理 |
|------|------|----------|
| 随机日期 | 固定震中位置，随机抽取同季节(±15天DOY)非震日期 | 控制空间+季节，仅检验时间特异性 |
| 时间平移 | 事件窗口整体平移+90/+180/+365天(避开真实地震) | 控制地点+季节周期，检验时间邻近性 |
| 随机位置 | 保持日期，震中在同环境区内随机平移到远离板块边界(>500km)区域 | 控制时间+环境类型，检验空间特异性 |
| 非震极端事件 | 台风/暴雨/火山(尤其海洋区)，使用EM-DAT/IBTrACS数据库 | 直接检验模型是否把极端天气误判为地震 |

**统计检验**：
- 对每类 placebo 重复 100+ 次，生成 MCC 零分布
- 真实事件 MCC 相对零分布的 Z-score 和 p 值
- **判定标准**：置换 p < 0.01
- 绘制：真实事件 vs 各类 placebo 的 MCC 箱线图

**失败应对**：
- 若 placebo MCC 也高 → 优先检查数据泄漏(T1) → 修复切分(T2) → 重跑
- 若修复后仍高 → 承认"弱关联/不可辨识"，转叙事为评估协议

**新增脚本**：`scripts/run_placebo.py`

---

#### T5. 噪声预算与时空聚合降噪论证（对应 MVS-1）

**目标**：从物理和数学上证明，虽然单像素效应量微弱(~1-3K)，但 WE-FTT 通过时空聚合可显著提升 SNR

**设计（三层论证）**：

**Layer 1：仪器噪声预算**
- 从 AMSR-2 官方文档提取各通道 NEDT (Noise Equivalent Delta Temperature)
  - 6.9 GHz: ~0.3K, 10.65 GHz: ~0.6K, 23.8 GHz: ~0.6K, 36.5 GHz: ~0.3K, 89.0 GHz: ~0.6K
- 定标不确定性(~1K)、地理配准误差传播

**Layer 2：格网化降噪传播**
- 0.25° 格网内包含 N_obs 个独立足迹观测
- σ_grid ≈ σ_instr / √N_obs（理论下界）
- 实测验证：在稳定靶区(撒哈拉/南极/副热带海洋)计算日际差分 TB 标准差

**Layer 3：Transformer 时空聚合**
- 模型在 batch_size=100,000 样本上训练
- 多头注意力机制的感受野覆盖多个格网点和多日时序
- 10 通道联合特征空间 → 有效观测维度 ≈ N_spatial × N_temporal × N_channels
- 在高维特征空间中，弱信号的统计可检测性随有效样本量提升

**输出**：
- 通道×环境区的 σ_eff 表（含理论和经验值）
- Z-score 分布图（异常效应量 / σ_eff）
- 新增方法论段落（~300 词 + 公式）

---

#### T6. 海洋区 ERA5 协变量条件化（对应 P4）

**目标**：引入 ERA5 海面气象数据做条件化分析，证明去除海况干扰后 FPR 下降

**数据获取**：
- ERA5 hourly single levels: 10m u-wind, 10m v-wind → 合成风速
- ERA5 SST, TCWV, Total precipitation
- 时空分辨率：0.25° × 1h → 聚合到与 AMSR-2 一致的日均/格网

**分析方案**：
```
方案1（硬性掩膜）：
  - 排除标准：风速 > 10 m/s 或 降水 > 5 mm/day
  - 重新计算 Zone A 的 FPR
  - 目标：FPR 从 0.242 降至 < 0.10

方案2（残差化）：
  - 对 Zone A 各通道 TB，以 SST/风速/TCWV 为自变量做回归
  - 使用 GAM 或 LightGBM 作为经验辐射传输模型
  - 在残差 TB 上重跑评估
  - 报告残差化前后的 MCC 和 FPR 对比

两种方案都做，对比报告
```

**新增脚本**：`scripts/era5_ocean_conditioning.py`

**输出**：ERA5 条件化代码 + 海洋 FPR 前后对比表/图

---

#### T7. 环境混杂条件化/残差化（对应 MVS-2）

**目标**：对陆地区(Zone B-E)做类似的环境协变量条件化

**协变量集**：
| 来源 | 变量 | 对应物理过程 |
|------|------|------------|
| ERA5-Land | Skin temperature, 2m temperature | 地表热状态 |
| ERA5-Land | Total precipitation, Snowfall | 降水/积雪 |
| GLDAS | Soil moisture (0-10cm) | 土壤含水量 |
| ERA5 | LAI (高/低植被) | 植被含水量 |
| ERA5 | TCWV | 大气水汽 |

**两条路线**：

路线1（残差化）：
- 对每通道 TB 做分区(Zone B-E)回归，用协变量拟合"正常环境驱动的 TB"
- 得到 TB_residual = TB_observed - TB_predicted
- 在 TB_residual 上重跑 WE-FTT pipeline（含关联规则挖掘+模型训练+评估）
- **关键指标**：残差化后 MCC 仍远离 0 且显著高于 placebo

路线2（条件化分层）：
- 按协变量分位数(25%/50%/75%)分层
- 在每层内做 EQ vs 非EQ 对比
- 分层 MCC + 分层置换检验
- Meta-analysis 合并各层效应量

**失败应对**：
- 若残差化后性能显著塌陷(MCC < 0.3)：承认模型主要在学环境因子
- 转贡献为"严谨评估协议 + 阴性结果的边界条件"（仍有学术价值）
- 缩小声明到"某些特定场景(如干旱裸地)"可能仍有微弱技能

---

### 第 3 阶段：TGRS 增强项（第 5-7 周）

#### T8. 半物理辐射传输敏感性分析（对应 GS-6）

**目标**：给出各通道对主要环境变量的理论灵敏度，让审稿人看到作者理解辐射传输物理

**陆地（τ-ω 模型）**：
```
TB_p = (1 - ω_p) × (1 - e^(-τ/cosθ)) × T_c
     + e_p(ε_soil) × e^(-τ/cosθ) × T_soil
     + T_sky × (1 - e_p(ε_soil)) × e^(-2τ/cosθ)

其中：
  ε_soil = f(SM, frequency)  → Dobson/Mironov 介电模型
  τ = f(VOD, frequency)       → 植被光学厚度
  ω = single-scattering albedo

一阶灵敏度：
  dTB/dSM ≈ de_p/dε × dε/dSM × T_soil × e^(-τ/cosθ)
  dTB/dVOD ≈ ...
  dTB/dTs ≈ ...
```

**海洋（Wentz/Meissner 模型）**：
```
e_p(SST, W, f) = Fresnel(ε(SST, salinity, f)) + Δe(W, f)

一阶灵敏度：
  dTB/dSST ≈ de/dSST × SST + e
  dTB/dW ≈ dΔe/dW × SST
```

**输出**：理论灵敏度表（通道×变量×环境区）+ 与实测残差异常的量级对比

---

#### T9. Table S5 重新包装

**核心策略**：化"自证不可测"为方法学核心动机

**当前问题**：Table S5 显示应力诱导的介电扰动仅能产生 0.4-2.8K 的 TB 变化，而土壤含水量变化可达 ~100K。Gemini 认为这是"自杀式证据"。

**重新包装的叙事**：

> "Table S5 presents a first-order Fresnel sensitivity comparison that highlights precisely why conventional single-pixel anomaly detection approaches have consistently failed in this domain. The estimated stress-induced TB perturbation (~0.4–2.8 K) is indeed comparable to single-footprint measurement noise (NEDT ~0.3–0.6 K) and dwarfed by environmental confounders (soil moisture alone can drive ~100 K changes). However, WE-FTT does not operate at the single-pixel level. By aggregating information across N ≈ 10^5 samples × 10 spectral channels in an environment-stratified feature space, the effective noise floor is reduced by 1–2 orders of magnitude (see Section X, Equation Y). This quantitative comparison therefore serves as the **methodological motivation** for our approach: because single-pixel SNR ≈ 1, advanced pattern recognition in high-dimensional feature spaces, guided by domain-specific association rules, becomes not merely advantageous but **necessary**."

**配套数学论证**（链接 T5 噪声预算）：
- σ_eff = σ_instr / √(N_obs × N_temporal)
- 在模型感受野内：N_eff ≈ 1000-10000（保守估计）
- 等效降噪后 σ_eff ≈ 0.01-0.03 K
- 此时 0.4-2.8 K 效应量对应 Z-score ≈ 13-280（高度显著）

---

#### T10. 公开可复现评估协议（对应 GS-11）

**内容清单**：
1. `evaluation_protocol/README.md` — 评估协议完整说明
2. `evaluation_protocol/data_splits/` — 事件级切分清单(JSON/CSV)
3. `evaluation_protocol/placebo/generate_placebos.py` — Placebo 生成脚本
4. `evaluation_protocol/era5/download_and_align.py` — ERA5 数据获取与对齐
5. `evaluation_protocol/run_evaluation.py` — 一键评估管线
6. `evaluation_protocol/reproduce_results.sh` — 结果复现脚本

**GitHub README 更新**：增加 "Reproducibility" 章节

---

### 第 4 阶段：论文改写与投递（第 7-8 周）

#### T11. 叙事全面重构

**建议标题**：
> "Near-Surface Microwave Anomaly Characterization Around Large Earthquakes: An Uncertainty-Propagated, Placebo-Controlled Evaluation Using Environment-Stratified Deep Learning"

**逐节改写要点**：

| 章节 | 改写方向 | 关键动作 |
|------|----------|----------|
| Title | 去"precursor detection" | 改为"anomaly characterization" |
| Abstract | 降维主张 | "earthquake-window-associated anomalies" + 强调placebo控制 |
| Introduction | 问题重定位 | "弱异常统计表征"而非"前兆检测" |
| Section 2 | 物理背景 | 补噪声预算+时空聚合降噪论证 |
| Section 3 | 方法 | 补事件级切分协议+权重生成防泄漏流程 |
| Section 3.x | Table S5 | 按T9重新包装为方法学动机 |
| Section 4 | 结果 | 前瞻验证提升为主结果; 补placebo图; 补ERA5条件化 |
| Discussion | 物理机制 | "possible surface coupling channel"，删除所有强因果措辞 |
| Discussion | 辐射传输 | 补τ-ω敏感性分析(T8) |
| Limitations | 扩充 | AMSR-2浅表限制、海洋高FPR、单传感器局限、M≥7.0适用边界 |
| Conclusion | 收缩 | 删除"reliable/near-perfect/operational early warning" |
| SI | 评估协议 | 补完整placebo协议+切分清单 |

**措辞搜索替换清单**：
```
"reliable detection" → "statistically significant association"
"near-perfect" → "strong" 或删除
"operational early warning" → "monitoring protocol"
"precursor" → "earthquake-window-associated anomaly" (大部分) / "possible precursor" (保守处)
"deep crustal stress" → 删除或"near-surface coupling"
"P-hole activation" → "possible charge carrier migration (Freund, 2011)" + caveats
"quasi-volumetric imaging" → 删除
```

#### T12. IEEE 双栏排版与投递

- 按 IEEE TGRS 模板 (`IEEEtran.cls`) 重排
- Cover Letter 核心要点：
  1. "如何在极强环境背景噪声中，通过深度学习+领域知识引导挖掘微弱遥感异常"
  2. 创新点：环境分区 + 知识引导权重 + placebo 控制评估协议
  3. 强调：前瞻验证(26事件) + 4类placebo(p<0.01) + ERA5条件化 + 公开代码
  4. **绝不提及 RSE 拒稿史**

---

## 四、关键风险与应对策略

| # | 风险 | 概率 | 影响 | 应对策略 |
|---|------|------|------|----------|
| 1 | 权重泄漏确认存在 | **高** | 致命 | P0最高优先审计；泄漏则修复pipeline后重跑全部实验。若修复后MCC显著下降，需重新评估论文可行性 |
| 2 | Placebo测试失败 | 低-中 | 致命 | 优先检查数据泄漏→修复切分→重跑。若仍失败则承认"弱关联/不可辨识"，转贡献为评估协议 |
| 3 | 残差化后性能塌陷 | 中 | 严重 | 把贡献转为"评估协议+阴性结果的边界条件"；仍有学术价值，可改投Earth and Space Science |
| 4 | 海洋条件化后FPR仍高 | 中 | 中等 | 把海洋区降级为"探索性分析"并明确标记限制 |
| 5 | TGRS审稿人要求跨传感器共证 | 中 | 中等 | "未来工作"中详细规划GNSS TEC/MODIS LST共证路线图。或退而投JSTARS |
| 6 | 修复泄漏后权重增强效果消失 | 低-中 | 严重 | 说明权重仍提供了"训练集内"的先验知识引导；即使增益减小也有方法学意义 |

---

## 五、验证清单（投递前必过）

| # | 检查项 | 通过标准 | 状态 |
|---|--------|----------|------|
| 1 | 权重泄漏审计 | 关联规则挖掘仅在训练折内执行，无跨集信息泄漏 | ⬜ |
| 2 | Placebo MCC零分布 | 4类placebo全部p<0.01 | ⬜ |
| 3 | 事件级切分性能 | MCC > 0.6，显著高于placebo零分布 | ⬜ |
| 4 | 区域留一(10°×10°) | 性能下降<20%（或明确标记失败区域） | ⬜ |
| 5 | 海洋ERA5条件化 | FPR显著下降（目标<0.10） | ⬜ |
| 6 | 陆地残差化TB | 残差化后技能仍显著（或转叙事为边界条件） | ⬜ |
| 7 | σ_eff噪声预算表 | 通道×环境区完整，异常效应量>2σ_eff | ⬜ |
| 8 | 措辞审查 | 全文无"reliable/near-perfect/operational early warning" | ⬜ |
| 9 | Table S5重新包装 | 叙事翻转为方法学动机，配套数学论证 | ⬜ |
| 10 | 评估协议公开 | GitHub包含完整可复现评估脚本 | ⬜ |

---

## 六、时间线概览

```
Week 1-2:  [T1] 权重泄漏审计 + [T2] 事件级切分实现
           ↓ 若发现泄漏：修复 + 重跑实验
Week 3-4:  [T4] Placebo实验 + [T5] 噪声预算
Week 4-5:  [T6] 海洋ERA5条件化 + [T7] 陆地残差化
Week 5-6:  [T8] 半物理辐射传输敏感性
Week 6-7:  [T9] Table S5重新包装 + [T10] 评估协议
Week 7-8:  [T11] 叙事重构 + [T12] IEEE排版 + 投递
```

**关键里程碑**：
- 🔴 Week 2 末：泄漏审计完成（决定是否需要重跑全部实验）
- 🟡 Week 4 末：Placebo结果出炉（决定论文是否仍可行）
- 🟢 Week 6 末：所有补强实验完成
- 🏁 Week 8 末：投递 TGRS
