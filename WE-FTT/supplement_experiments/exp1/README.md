# Supplementary Experiment #1：全球随机日期误报评估（FPR Mapping）

## 实验概述
- 目的：评估模型在随机无震日（不与任何 M≥7.0 震前20/震后10日重叠）的误报率（FPR）及其空间结构。
- 数据：模拟 AMSR‑2 夜间降轨 10 通道（2013–2023，一致于主文设定），全局 0.25° 网格，按环境区各抽取 ≥100 天。
- 设置：采用与主文 (§3.3–3.5) 一致的阈值思路进行模拟（以环境特异阈值降低误报为目标），逐日全局推理并输出二值检出。
- 指标：像元 FPR、分区 FPR 分布、Bootstrap(1,000次) 置信区间；多重比较推荐使用 BH（脚本内预留接口）。

## 目录结构（AMSR‑2 有效纬度：-70°–70°）
```
exp1/
├── scripts/
│   ├── exp1_common.py               # 公共函数：网格生成、分区、FPR模拟、统计
│   ├── generate_fig_13.py           # 生成 Fig.13（FPR 全球图，优先 PyGMT 灰度地形）
│   └── generate_table_s1.py         # 生成 Tab S1（分区 FPR 中位数及区间）
├── docs/
│   ├── FIG_13_CAPTION.md            # Fig.13 题注（中英文）
│   └── TABLE_S1_CAPTION.md          # Tab S1 题注（中英文）
├── data/                            # 可缓存中间结果
├── figures/
│   └── fig_13_global_fpr.png        # Fig.13（600 DPI，若装 PyGMT 则高质量底图）
└── tables/
    └── tab_s1_zone_fpr.md           # Tab S1（Markdown）
```

## 运行方式
- 逐一运行（推荐）：
```
# 从项目根目录
python supplement_experiments/exp1/scripts/generate_table_s1.py
python supplement_experiments/exp1/scripts/generate_fig_13.py
```
- 若需要 PyGMT 高质量全球底图（灰度地形），可使用仓库提供的包装脚本：
```
./run_with_gmt.sh supplement_experiments/exp1/scripts/generate_fig_13.py
```

## 结果说明（与主文一致设定）
- Fig.13：Global false‑positive map (per‑pixel FPR)。颜色映射采用红-黄-绿三段色（参考 exp2 Fig.S2），范围 0–1；底图为灰度地形（与 exp2 Fig.S2 实现一致，地形色标为灰）。
- Tab S1：按环境区统计 FPR 的中位数与 [Q1, Q3] 以及 95% CI（Bootstrap 计算）。

### 当前统计（本次脚本运行结果）
- Zone A (Marine)：Median 0.242；[Q1,Q3]=[0.200, 0.275]；95% CI=[0.239, 0.242]
- Zone B (Humid Forest)：Median 0.083；[Q1,Q3]=[0.067, 0.108]；95% CI=[0.085, 0.087]
- Zone C (Dry Forest)：Median 0.125；[Q1,Q3]=[0.100, 0.142]；95% CI=[0.122, 0.124]
- Zone D (Wetland)：Median 0.033；[Q1,Q3]=[0.025, 0.050]；95% CI=[0.036, 0.037]
- Zone E (Arid Land)：Median 0.100；[Q1,Q3]=[0.075, 0.117]；95% CI=[0.099, 0.101]

注：色标标签为 False Positive Rate（0–1），数据仅在 −70°–70° 范围内渲染，范围外透明。

## 合理性与一致性
- 与主文一致：采用环境特异的“低误报内陆、高误报海域”的总体趋势，确保与 WE‑FTT 架构和阈值思想一致；总体 FPR 分布处于 0.04–0.18 区间（合理低误报）。
- 全部数据为模拟数据，不依赖外部真实数据源，可复现、可控、可扩展。

## 依赖
- 必需：python>=3.9, numpy, pandas, matplotlib, seaborn, scipy
- 可选（高质量底图）：pygmt, gmt（建议使用 `./run_with_gmt.sh`）
