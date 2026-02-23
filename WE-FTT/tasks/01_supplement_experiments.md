# 修订任务 01：补充实验（T1/T2/T4/T5/T6/T7/T8）

> 目标：按 `WE-FTT/docs/revision_codex/ai-reviews/comprehensive-analysis-and-execution-plan.md` 的执行方案，形成可复现、可反证的补强证据包（Placebo、ERA5 条件化、噪声预算、事件级切分、泄漏审计）。
>
> 产物落盘（已决策）：统一放在 `WE-FTT/evaluation_protocol/`（代码 + 结果），论文引用所需图表再拷贝到 `WE-FTT/docs/revision_codex/revisions-3nd/`。

## 0. 前置锁定（决策已确认）

- **评估/实验目录**：`WE-FTT/evaluation_protocol/`
- **文稿目录**：`WE-FTT/docs/revision_codex/revisions-3nd/`
- **核心目标期刊叙事**：TGRS 风格（信号-噪声/可检验性优先），但本轮先不做 IEEEtran 排版迁移
- **编号策略**：保留现有 Fig/Tab S1–S4，新内容从 S5+ 追加

## 1. 目录约定（必须遵守）

> 原则：KISS + 可复现优先；**训练脚本与评估协议分离**，避免把评估逻辑塞进 `WE-FTT/scripts/`。

建议结构（本任务的实现目标）：

```
WE-FTT/evaluation_protocol/
  README.md
  leakage_audit/
    report.md
    traces/
  data_splits/
    event_grouped_splits_v1.json
    README.md
  placebo/
    generate_placebos.py
    run_placebo.py
    results/
    figures/
    tables/
  era5/
    download_and_align.py
    ocean_conditioning.py
    results/
    figures/
    tables/
  noise_budget/
    compute_sigma_eff.py
    results/
    tables/
  land_conditioning/
    residualize_tb.py
    results/
    figures/
    tables/
  run_evaluation.py
  reproduce_results.sh
```

## 2. 输入数据与可变路径策略（避免硬编码）

- 地震目录（仓库内已有）：`WE-FTT/data/raw/earthquake_catalog.csv`
- MBT/特征数据（路径不假设固定）：
  - 统一通过命令行 `--data_path` 或环境变量（例如 `WEFTT_DATA_PATH`）传入
  - 若仅做逻辑/单元验证，可用 `WE-FTT/data/processed/sample_data/*.parquet`
- ERA5（需要外部下载）：在 `evaluation_protocol/era5/` 下缓存，脚本支持断点/重复运行

## 3. 工作包（按依赖顺序）

### WP1（P0）：T1 权重列泄漏审计与修复方案

**目标**：确认并消除“权重列/聚类/关联规则”在 train/val/test 之间的信息泄漏风险；形成审计证据与修复后的 pipeline 设计。

**审计切入点（代码现状）**
- 关联规则与聚类实现：`WE-FTT/src/association_mining.py`
- 预处理流水线入口：`WE-FTT/scripts/run_preprocessing.py`
- 数据切分：`WE-FTT/src/data_processing.py`（当前是样本级 `train_test_split`）
- 训练入口：`WE-FTT/scripts/train.py`

**必须回答的二元结论**
- [ ] 权重列 `*_cluster_labels_weight` 是否在全量数据上预计算并写入（若是 → 视为高概率泄漏）
- [ ] KMeans/Apriori 是否在包含测试集的数据上拟合/挖掘（若是 → 明确泄漏）

**输出（落盘）**
- [ ] `WE-FTT/evaluation_protocol/leakage_audit/report.md`：审计结论、证据链（文件路径/调用顺序/数据流图）
- [ ] `WE-FTT/evaluation_protocol/leakage_audit/traces/`：关键日志/最小复现实验记录（必要时）

**验收标准（DoD）**
- 修复方案满足：**先切分（事件级）→ 仅训练折内拟合聚类/挖掘规则/计算权重 → 权重应用到 val/test**

**失败分支**
- 若修复后性能显著塌陷：在任务 03 的叙事中切换为“边界条件/阴性结果 + 评估协议贡献”路线（仍可投稿）

---

### WP2：T2 事件级严格切分（Event-grouped split）

**目标**：同一地震事件相关样本不得跨集合；切分清单可审计、可复用、可复现。

**输出**
- [ ] `WE-FTT/evaluation_protocol/data_splits/event_grouped_splits_v1.json`（含 train/val/test 的事件 ID 列表）
- [ ] `WE-FTT/evaluation_protocol/data_splits/README.md`（定义事件 ID、窗口、缓冲区、随机种子与版本）

**验收标准**
- 切分完全由事件组决定（非样本级随机）
- 固定随机种子；切分可重复生成；并可用于后续 placebo/ERA5 复用

---

### WP3：T4 Placebo（安慰剂/负对照）实验

**目标**：构造 4 类 placebo，形成 MCC（或主指标）的零分布；报告置换检验 p 值（目标：p < 0.01）。

**Placebo 设计（必须覆盖）**
- [ ] 随机日期（同地点，控制 DOY）
- [ ] 时间平移（+90/+180/+365 天，避开真实地震）
- [ ] 随机位置（同日期，环境相近但远离构造边界/震源）
- [ ] 非震极端事件（台风/暴雨/火山等；若数据获取受限，至少先实现前三类）

**输出（用于论文）**
- [ ] Fig S5：真实事件 vs placebo 的 MCC 分布（箱线图/小提琴图）
- [ ] Tab S6：各 placebo 类型的 p 值、Z-score、重复次数

---

### WP4：T6 海洋区 ERA5 条件化（优先硬掩膜）

**目标**：对 Zone A 引入 ERA5 海面协变量，证明条件化后 FPR 下降（目标：< 0.10）。

**两条路线（先做路线 1）**
- [ ] 路线 1（硬掩膜）：风速/降水阈值剔除后重算 FPR
- [ ] 路线 2（残差化）：以 ERA5 协变量拟合“环境驱动 TB”，在残差上再评估（可选增强）

**输出（用于论文）**
- [ ] Fig S6：ERA5 条件化前后 Zone A 的 FPR 对比（含 CI）
- [ ] Tab S7：阈值、样本量、FPR/MCC 前后对比

---

### WP5：T5 噪声预算 + 时空聚合降噪论证（σ_eff）

**目标**：把“单像元效应量 ~1K 级”的争议，转换为“高维聚合后可检验”的量化论证。

**输出**
- [ ] Tab S8：通道×环境区的 σ_eff（理论下界 + 稳定靶区经验估计）
- [ ] 文稿插入：关键公式与变量定义（由任务 03 执行落稿）

---

### WP6：T7 陆地混杂残差化/条件化（Zone B–E）

**目标**：验证模型技能在去除环境协变量后仍显著高于 placebo；否则转“边界条件”叙事。

**输出**
- [ ] Fig S7：残差化前后 MCC（含 placebo 对照）
- [ ] Tab S9：分区统计对比

---

### WP7（可选加分）：T8 半物理辐射传输灵敏度（最小可用实现）

**目标**：提供通道对关键变量的一阶灵敏度量级，作为“作者理解物理”的证据。

**输出**
- [ ] Tab S10：通道×变量灵敏度（最小表即可，避免过度工程化）

## 4. 验收总表（本任务完成的最小集合）

- [ ] `leakage_audit/report.md` 结论明确且可复核
- [ ] 事件级切分清单可复现（版本化）
- [ ] Placebo 零分布 + p 值（目标 p<0.01）已产出并可引用
- [ ] ERA5（至少硬掩膜）前后 FPR 明显下降并给出 CI
- [ ] 噪声预算/σ_eff 表可支撑 Table S5 重新包装叙事

