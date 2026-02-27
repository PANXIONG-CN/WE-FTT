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
- [x] 权重列 `*_cluster_labels_weight` 是否在全量数据上预计算并写入（结论：**是**，存在高概率泄漏风险）
- [x] KMeans/Apriori 是否在包含测试集的数据上拟合/挖掘（结论：当前复现路径无法从训练入口排除，按审计链视为**明确泄漏风险**）

**输出（落盘）**
- [x] `WE-FTT/evaluation_protocol/leakage_audit/report.md`：审计结论、证据链（文件路径/调用顺序/数据流图）
- [x] `WE-FTT/evaluation_protocol/leakage_audit/traces/`：关键日志/最小复现实验记录（必要时）
- [x] `WE-FTT/scripts/train.py`：主训练链已统一为泄漏安全流程（事件级切分 + 仅 train 拟合标准化/权重，再应用 val/test）
- [x] `WE-FTT/results/smoke_train_safe_v2/random_forest/data_pipeline_meta.json`：冒烟验证记录（`split_mode=event_grouped`，`foldwise_weights_enabled=true`）

**验收标准（DoD）**
- 修复方案满足：**先切分（事件级）→ 仅训练折内拟合聚类/挖掘规则/计算权重 → 权重应用到 val/test**

**失败分支**
- 若修复后性能显著塌陷：在任务 03 的叙事中切换为“边界条件/阴性结果 + 评估协议贡献”路线（仍可投稿）

---

### WP2：T2 事件级严格切分（Event-grouped split）

**目标**：同一地震事件相关样本不得跨集合；切分清单可审计、可复用、可复现。

**输出**
- [x] `WE-FTT/evaluation_protocol/data_splits/event_grouped_splits_v1.json`（含 train/val/test 的事件 ID 列表）
- [x] `WE-FTT/evaluation_protocol/data_splits/README.md`（定义事件 ID、窗口、缓冲区、随机种子与版本）
- [x] 区域留一外推评估产物（v1/v2/v3）：
  - `WE-FTT/evaluation_protocol/data_splits/results/spatial_cv_summary_v1.json`
  - `WE-FTT/evaluation_protocol/data_splits/results/spatial_cv_summary_v2_hgb_daymean.json`
  - `WE-FTT/evaluation_protocol/data_splits/results/spatial_cv_summary_v3_hgb_daymean_zoneThr.json`
  - `WE-FTT/evaluation_protocol/data_splits/results/spatial_cv_summary_v3_hgb_pixel_landResid_zoneThr.json`

**验收标准**
- 切分完全由事件组决定（非样本级随机）
- 固定随机种子；切分可重复生成；并可用于后续 placebo/ERA5 复用

---

### WP3：T4 Placebo（安慰剂/负对照）实验

**目标**：构造 4 类 placebo，形成 MCC（或主指标）的零分布；报告置换检验 p 值（目标：p < 0.01）。

**Placebo 设计（必须覆盖）**
- [x] 随机日期（同地点，控制 DOY）
  - 结果：`p_value_ge=0.00893`，`z=3.5448`，`n_repeats_used=111`（`zone_filter=land`，`aggregation=day_mean`）
  - JSON：`WE-FTT/evaluation_protocol/placebo/results/random_date_results_land_daymean_v8.json`
  - Fig：`WE-FTT/evaluation_protocol/placebo/figures/fig_s5_mcc_random_date_v8.png`
  - Tab：`WE-FTT/evaluation_protocol/placebo/tables/tab_s6_placebo_random_date_v8.csv`
  - Plan：`WE-FTT/evaluation_protocol/placebo/plans/random_date_plan_v2.jsonl`
- [x] 时间平移（+90/+180/+365 天，避开真实地震）
  - 结果：`p_value_ge=0.00752`，`z=3.2726`，`n_repeats_used=398`（`zone_filter=land`，`aggregation=day_mean`）
  - JSON：`WE-FTT/evaluation_protocol/placebo/results/time_shift_results_land_daymean_v8.json`
  - Fig：`WE-FTT/evaluation_protocol/placebo/figures/fig_s5_mcc_time_shift_v8.png`
  - Tab：`WE-FTT/evaluation_protocol/placebo/tables/tab_s6_placebo_time_shift_v8.csv`
  - Plan：`WE-FTT/evaluation_protocol/placebo/plans/time_shift_plan_v2.jsonl`
- [x] 随机位置（同日期，环境相近但远离构造边界/震源）
  - 结果：`p_value_ge=0.00249`，`z=2.7990`，`n_repeats_used=400`（`zone_filter=land`，`aggregation=day_mean`）
  - JSON：`WE-FTT/evaluation_protocol/placebo/results/random_location_results_land_daymean_v9.json`
  - Fig：`WE-FTT/evaluation_protocol/placebo/figures/fig_s5_mcc_random_location_v9.png`
  - Tab：`WE-FTT/evaluation_protocol/placebo/tables/tab_s6_placebo_random_location_v9.csv`
  - Plan：`WE-FTT/evaluation_protocol/placebo/plans/random_location_plan_v3.jsonl`（donor 约束：同半球 + `|Δlat|<=20°`；不足时自动放宽并写入 plan）
- [x] 非震极端事件（ERA5 代理）
  - 结果：`p_value_ge=0.00249`，`z=7.0195`，`n_repeats_used=400`（`zone_filter=land`，`aggregation=day_mean`）
  - JSON：`WE-FTT/evaluation_protocol/placebo/results/nonseismic_extreme_results_land_daymean_v1.json`
  - Fig：`WE-FTT/evaluation_protocol/placebo/figures/fig_s5_mcc_nonseismic_extreme_v1.png`
  - Tab：`WE-FTT/evaluation_protocol/placebo/tables/tab_s6_placebo_nonseismic_extreme_v1.csv`
  - Plan：`WE-FTT/evaluation_protocol/placebo/plans/nonseismic_extreme_plan_v1.jsonl`

**输出（用于论文）**
- [x] Fig S5：真实事件 vs placebo 的 MCC 分布（箱线图/小提琴图）
  - `WE-FTT/evaluation_protocol/placebo/figures/fig_s5_mcc_random_date_v8.png`
  - `WE-FTT/evaluation_protocol/placebo/figures/fig_s5_mcc_time_shift_v8.png`
  - `WE-FTT/evaluation_protocol/placebo/figures/fig_s5_mcc_random_location_v9.png`
  - `WE-FTT/evaluation_protocol/placebo/figures/fig_s5_mcc_nonseismic_extreme_v1.png`
- [x] Tab S6：各 placebo 类型的 p 值、Z-score、重复次数
  - `WE-FTT/evaluation_protocol/placebo/tables/tab_s6_placebo_random_date_v8.csv`
  - `WE-FTT/evaluation_protocol/placebo/tables/tab_s6_placebo_time_shift_v8.csv`
  - `WE-FTT/evaluation_protocol/placebo/tables/tab_s6_placebo_random_location_v9.csv`
  - `WE-FTT/evaluation_protocol/placebo/tables/tab_s6_placebo_nonseismic_extreme_v1.csv`

**复现参数（锁定）**
- `zone_filter=land`，`aggregation=day_mean`
- `thr_select_mode=fixed`，`thr_fixed=0.5`
- `pre_days=20`，`post_days=10`
- `pixels_per_event_day=100`，`control_dates_per_event=1`，`min_rep_n=200`，`seed=42`

---

### WP4：T6 海洋区 ERA5 条件化（优先硬掩膜）

**目标**：对 Zone A 引入 ERA5 海面协变量，证明条件化后 FPR 下降（目标：< 0.10）。

**两条路线（先做路线 1）**
- [x] 路线 1（硬掩膜）：风速/降水阈值剔除后重算 FPR（采用 `wind=10m/s, precip=5mm/day`，并用 TP 目标选阈值；`FPR 0.0906 -> 0.0631`）
- [x] 路线 2（残差化）：以 ERA5 协变量拟合“环境驱动 TB”，在残差上再评估（已执行；`FPR` 下降但 `MCC` 为负，作为附加边界结果保留）

**输出（用于论文）**
- [x] Fig S6：ERA5 条件化前后 Zone A 的 FPR 对比（含 CI）
  - `WE-FTT/evaluation_protocol/era5/figures/fig_s6_era5_ocean_fpr_v2_fixed_w10_p5_tp.png`
  - `WE-FTT/evaluation_protocol/era5/figures/fig_s6_era5_ocean_residualize_geo_v1.png`（附加）
- [x] Tab S7：阈值、样本量、FPR/MCC 前后对比
  - `WE-FTT/evaluation_protocol/era5/tables/tab_s7_era5_ocean_mask_v2_fixed_w10_p5_tp.csv`
  - `WE-FTT/evaluation_protocol/era5/tables/tab_s7_era5_ocean_residualize_geo_v1.csv`（附加）

---

### WP5：T5 噪声预算 + 时空聚合降噪论证（σ_eff）

**目标**：把“单像元效应量 ~1K 级”的争议，转换为“高维聚合后可检验”的量化论证。

**输出**
- [x] Tab S8：通道×环境区的 σ_eff（理论下界 + 稳定靶区经验估计）
  - `WE-FTT/evaluation_protocol/noise_budget/tables/tab_s8_sigma_eff_v2.csv`
  - `WE-FTT/evaluation_protocol/noise_budget/results/sigma_eff_v2.meta.json`
  - 论文汇总表：`WE-FTT/docs/revision_codex/revisions-3nd/tab_s8_sigma_eff_summary.csv`
  - 三口径结果：全局经验口径 `mean_z=0.041~0.121`；同事件经验口径 `mean_z=1.281~2.047`；配对稳健经验口径 `mean_z=5.971~13.755`
  - 严格验收（每通道）：配对稳健经验口径下 `criterion_gt_2sigma_empirical_paired_robust` 为 `50/50`（最小 `z=2.720`）
- [x] 文稿插入：关键公式与变量定义（由任务 03 执行落稿）

---

### WP6：T7 陆地混杂残差化/条件化（Zone B–E）

**目标**：验证模型技能在去除环境协变量后仍显著高于 placebo；否则转“边界条件”叙事。

**输出**
- [x] Fig S7：残差化前后 MCC（含 placebo 对照）
  - `WE-FTT/evaluation_protocol/land_conditioning/figures/fig_s7_land_residualize_v1.png`
  - `WE-FTT/evaluation_protocol/land_conditioning/figures/fig_s7_land_residualize_v2.png`（改进口径：val 选阈值 + zone-wise 背景 + ERA5 协变量）
- [x] Tab S9：分区统计对比
  - `WE-FTT/evaluation_protocol/land_conditioning/tables/tab_s9_land_residualize_v1.csv`
  - `WE-FTT/evaluation_protocol/land_conditioning/tables/tab_s9_land_residualize_v2.csv`
  - 改进版（v2）主结果：`real_mcc_resid=0.0997`，`p_value_resid_ge=0.00498`（`land_all`）；满足“残差后仍显著高于 placebo”
  - 说明：v1 仍保留为对照（固定阈值口径下的阴性结果），用于边界条件讨论

---

### WP7（可选加分）：T8 半物理辐射传输灵敏度（最小可用实现）

**目标**：提供通道对关键变量的一阶灵敏度量级，作为“作者理解物理”的证据。

**输出**
- [x] Tab S10：通道×变量灵敏度（最小表即可，避免过度工程化）
  - `WE-FTT/evaluation_protocol/radiative_sensitivity/tables/tab_s10_fresnel_sensitivity_v1.csv`
  - `WE-FTT/evaluation_protocol/radiative_sensitivity/results/fresnel_sensitivity_v1.meta.json`

## 4. 验收总表（本任务完成的最小集合）

- [x] `leakage_audit/report.md` 结论明确且可复核
- [x] 事件级切分清单可复现（版本化）
- [x] Placebo 零分布 + p 值（目标 p<0.01）已产出并可引用
- [x] ERA5（至少硬掩膜）前后 FPR 明显下降并给出 CI（`0.0906 -> 0.0631`，见 `tab_s7_era5_ocean_mask_v2_fixed_w10_p5_tp.csv`）
- [x] 噪声预算/σ_eff 表可支撑 Table S5 重新包装叙事
- [x] 陆地残差化实验已完成：改进版（v2）满足残差后显著；v1 阴性结果作为边界条件补充
- [x] Tab S10 半物理灵敏度最小表已产出
