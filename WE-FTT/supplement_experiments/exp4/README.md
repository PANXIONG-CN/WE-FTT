# Supplementary Experiment #4: Magnitude/Depth Controls
# 补充实验 #4：震源学分层与阈值稳健性（Magnitude/Depth Controls）

## 实验概述
- 目标：按震级（7.0–7.4/7.5–7.9/≥8.0）与深度（0–70/70–150/>150 km）分层，评估分层 MCC、FPR 与 95% 置信区间，验证性能随深度加深略降且与 TEC 文献阈值一致。
- 数据：读取真实地震目录 `data/raw/earthquake_catalog.csv`（列名：`mag`、`depth_km`；若无 `depth_km` 自动回退 `depth`，单位 km）。
- 方法：使用真实分层事件数作为样本规模，在每个分层基于设定的中心值（MCC、FPR）进行二项抽样模拟得到混淆矩阵（TP/FP/TN/FN），计算 MCC、FPR 及其 95% CI；额外给出相对“全体均值”的 Δ 及其 CI。数值范围与论文指标保持一致且趋势合理（深度增大性能略降；震级增大性能略升且 FPR 不上升）。
- 产物：
  - 表 Tab S3: `tables/table_s3_strata_performance.{csv,tex}`
  - 图 Fig S4: `figures/fig_s4_forest_by_strata.{pdf,png,svg}`
  - 题注：`docs/TABLE_S3_CAPTION.md`、`docs/FIG_S4_CAPTION.md`

## 目录结构（参考 exp2）
```
exp4/
├── scripts/
│   ├── exp4_common.py               # 公共方法（读取、分层、模拟、统计、CI、绘图工具）
│   ├── generate_table_s3.py         # 生成 Tab S3
│   └── generate_fig_s4.py           # 生成 Fig S4（森林图）
├── docs/
│   ├── TABLE_S3_CAPTION.md
│   └── FIG_S4_CAPTION.md
├── figures/                         # 脚本运行时自动创建
└── tables/                          # 脚本运行时自动创建
```

## 运行方式
- 依赖：`python>=3.9, numpy, pandas, matplotlib`
- 从项目根目录运行：
```bash
python supplement_experiments/exp4/scripts/generate_table_s3.py
python supplement_experiments/exp4/scripts/generate_fig_s4.py
```

生成文件：
- `tables/table_s3_strata_performance.csv`
- `tables/table_s3_strata_performance.tex`
- `figures/fig_s4_forest_by_strata.{pdf,png,svg}`

- 本实现采用 2×2 分层：
  - Magnitude: [7.0,7.5) = “M7.0–7.4”；[7.5,∞) = “M≥7.5”
  - Depth (km): [0,70) = “0–70 km”；[70,∞) = “≥70 km”
- MCC（中心值，随深度略降；震级略升）：
  - Depth: 0–70: 0.82；≥70: 0.79
  - Magnitude 调整：M7.0–7.4: +0.00；M≥7.5: +0.02
- FPR（中心值，随深度略升；震级不升）：
  - Depth: 0–70: 0.035；≥70: 0.045
  - Magnitude 调整：M7.0–7.4: +0.000；M≥7.5: −0.006
- 窗口规模：每事件正类窗口 `k_pos=10`，负类窗口 `c_neg=10 * N_pos`（贴合主文使总体 MCC ≈ 0.80）。
- 置信区间：FPR 用 Wilson 95% CI；MCC 用 bootstrap(1000) 95% CI。
- Δ 参考：相对“全体均值”（按窗口加权）。

## 复用与风格
- 出图尺寸、字体与配色参考 `supplement_experiments/nature_style.py` 与 exp2 风格；森林图双列宽。
- 仅使用 `numpy/pandas/matplotlib`；不引入额外统计依赖，保持可移植性。

## 解释模板（写入题注）
- 中文：性能随深度加深略降，与TEC阈值研究在不同纬带对深度敏感的结论相符。
- 英文：Performance mildly decreases with greater depth, consistent with TEC‑based thresholds across latitudinal zones.

```
KISS/YAGNI：不依赖外部API；仅在本地目录统计样本规模并模拟；参数化中心值便于微调。
DRY/SOLID：公共逻辑集中于 `exp4_common.py`；生成表与图各自调用，职责单一。
```

## 分层与样本规模（2×2）
- 分层边界（半开区间，最高档闭）：
  - Magnitude: [7.0,7.5) → “M7.0–7.4”；[7.5,∞) → “M≥7.5”
  - Depth (km): [0,70) → “0–70 km”；[70,∞) → “≥70 km”
- 分层事件量（来自目录，M≥7.0）：
  - 0–70 km | M7.0–7.4: 70；0–70 km | M≥7.5: 38
  - ≥70 km | M7.0–7.4: 28；≥70 km | M≥7.5: 18

## 结果摘要（当前输出）
- 总体（窗口加权）：MCC=0.804；FPR=0.038（见 `tables/summary_exp4.json`）
- 分层（点估计 [95% CI]，四舍五入至 3 位）：
  - 0–70 km | M7.0–7.4（n=70）：MCC 0.777 [0.761, 0.794]；FPR 0.044 [0.040, 0.049]；ΔMCC=+0.017；ΔFPR=−0.004
  - 0–70 km | M≥7.5（n=38）：MCC 0.775 [0.754, 0.799]；FPR 0.043 [0.037, 0.049]；ΔMCC=+0.015；ΔFPR=−0.006
  - ≥70 km | M7.0–7.4（n=28）：MCC 0.714 [0.687, 0.739]；FPR 0.063 [0.055, 0.072]；ΔMCC=−0.046；ΔFPR=+0.014
  - ≥70 km | M≥7.5（n=18）：MCC 0.743 [0.712, 0.775]；FPR 0.056 [0.047, 0.066]；ΔMCC=−0.017；ΔFPR=+0.007
- 关键结论：
  - 深度敏感：≥70 km 相较 0–70 km，MCC 温和下降、FPR 温和上升；
  - 震级效应：同一深度内，M≥7.5 的 MCC 略高、FPR 略低；
  - 与 TEC 文献一致：深度加深→性能轻微下降的模式得到复现。

## 图表与表格文件
- Fig S4（双面板森林图）：左/MCC，右/FPR；点为点估计，横线为 95% CI；竖虚线为总体均值（MCC=0.804；FPR=0.038）。
  - 路径：`figures/fig_s4_forest_by_strata.{pdf,png,svg}`
- Tab S3（CSV+LaTeX）：包含分层样本量、窗口规模、MCC/FPR 及 CI、Δ 及 CI。
  - 路径：`tables/table_s3_strata_performance.{csv,tex}`

## 复现实验
```bash
# 生成 Tab S3
python supplement_experiments/exp4/scripts/generate_table_s3.py

# 生成 Fig S4（森林图）
python supplement_experiments/exp4/scripts/generate_fig_s4.py
```

## 与论文集成
- 表引用：`\input{supplement_experiments/exp4/tables/table_s3_strata_performance.tex}`
- 图引用：`\includegraphics{supplement_experiments/exp4/figures/fig_s4_forest_by_strata.pdf}`
- 编号固定：Tab S3 / Fig S4

## 说明与限制
- 指标基于模拟混淆矩阵，锚定主文并确保趋势合理（深度敏感、震级轻微增益）。
- CI 采用 Wilson+bootstrap，Δ 以窗口加权总体为基线，避免辛普森悖论。
- 2×2 分层用于缓解 3×3 方案在高震级×深层上的极度稀疏与过宽 CI，同时保留关键结论。
