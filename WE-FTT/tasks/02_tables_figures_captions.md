# 修订任务 02：表格/图件与题注（新增 + 修改）

> 目标：把补强实验的产物（表/图）以“最小改动、最大可审计性”的方式接入 `revisions-3nd`，并按 TGRS 风格统一题注与措辞，避免 RSE 拒稿点复现。
>
> 参考方案：`WE-FTT/docs/revision_codex/ai-reviews/comprehensive-analysis-and-execution-plan.md`（T5/T6/T9/T11）。

## 0. 现状清点（来自 `revisions-2nd/main_rev.tex`）

### 主文图（当前已引用的文件名）
- `image1.png`（Fig 1）
- `image2.png`（Fig 2）⚠️ 题注含 *near-perfect reliability*
- `image3.png`（Fig 3）
- `image4.png`（Fig 4）
- `image5.png`（Fig 5）
- `image6.png`（Fig 6）
- `image7.png`（Fig 7）
- `image8.png`（Fig 8）
- `image9.png`（Fig 9）
- `image10.png`（Fig 10）
- `image11.png`（Fig 11）
- `image12.png`（Fig 12）
- `fig_13_global_fpr.pdf`（Fig 13）

### 补充材料（当前已引用的文件名）
- `fig_s1_temporal_evolution.pdf`（Fig S1）
- `fig_s2_spatial_coverage.png`（Fig S2）
- `fig_s3_marine_controls.pdf`（Fig S3）
- `fig_s4_forest_by_strata.pdf`（Fig S4）
- Table S1–S4（均在 tex 内直接写表格环境）

## 1. 编号策略（已锁定，禁止自行改动）

- 保留现有 Fig S1–S4、Tab S1–S4，不做重编号
- 新增图表从 **S5** 开始顺延：Fig S5、Tab S5、…
- 若必须把某项“提升为主结果”：优先通过 **正文强化叙事 + 引用现有补充图** 达成，避免触发全稿重编号雪崩

## 2. 文件命名与落盘规则（必须遵守）

### 2.1 产物源头
- 实验脚本产物统一生成在：`WE-FTT/evaluation_protocol/**/{figures,tables,results}/`

### 2.2 论文引用目录
- 最终被 `\includegraphics{...}` 引用的文件，**拷贝到**：`WE-FTT/docs/revision_codex/revisions-3nd/`
- 目标：`revisions-3nd/main_clean.tex` 单目录可编译（不依赖 `supplement_experiments/` 的相对路径）

### 2.3 建议命名（与编号绑定）
- Placebo：`fig_s5_placebo_mcc_distribution.pdf`
- ERA5 海洋条件化：`fig_s6_era5_ocean_conditioning.pdf`
- 残差化对照：`fig_s7_residualization_vs_placebo.pdf`
- 噪声预算表：`tab_s8_sigma_eff.tex`（或直接写入 tex；二选一，优先可复用的 `\input{}`）
- Table S5（Fresnel/介电灵敏度重新包装）：`tab_s5_fresnel_sensitivity.tex`

> 注意：命名一旦进入 `revisions-3nd/` 并被引用，后续不得随意改名（避免引用断裂）。

## 3. 题注与措辞统一规则（核心）

### 3.1 禁用/高风险词（按 T11）
- 禁用：`near-perfect`、`reliable detection`、`operational early warning`、过强因果词（`strongly support` 等）
- 优先替换为：`statistically significant association`、`earthquake-window-associated anomaly`、`screening/evaluation protocol`

### 3.2 已知需要立刻改的题注（最小集合）
- [x] Fig 2 当前题注：`Environment-specific precursor signatures achieving near-perfect reliability.`
  - 必须改为保守表述（示例方向）：环境分区下的 *zone-dependent anomaly patterns / association signatures*（不承诺“可靠预报”）

### 3.3 题注内容必须包含的信息
对所有新增 Fig/Tab（S5+）：
- 指标定义（MCC/FPR/p 值/CI 的计算口径）
- 样本量与重复次数（placebo 的 N）
- 显著性阈值（例如 p<0.01）
- 关键阈值（ERA5 掩膜阈值/变量定义）

## 4. 样式与格式（最小可用，不做过度工程化）

- 优先矢量：`*.pdf`（曲线/箱线图/森林图）
- 必须统一字体/字号（可复用 `WE-FTT/supplement_experiments/nature_style.py` 的风格参数，但不要强依赖该目录路径）
- 轴标签含单位；色标范围明确；避免“仅凭颜色猜含义”

## 5. 验收标准（DoD）

- [x] `revisions-3nd/main_clean.tex` 中所有新增图表引用路径存在且可编译
- [x] 新增 S5+ 图表题注均满足 3.3 要求
- [x] 全文与题注通过“禁用词扫描”（见任务 03 的清单）
