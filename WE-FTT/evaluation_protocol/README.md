# evaluation_protocol（可复现评估协议｜工作区）

本目录用于承载 **WE-FTT 转投/修订**所需的可复现评估协议与补强实验（与训练脚本解耦）。

## 设计目标

- **可审计**：每个结论能追溯到脚本、参数、数据切分清单与产物文件
- **可复现**：在同一数据输入下可重复产出一致的表/图
- **最小耦合**：不把评估逻辑混入 `WE-FTT/scripts/` 训练入口

## 任务入口

- 任务拆解与验收标准见：
  - `WE-FTT/tasks/01_supplement_experiments.md`
  - `WE-FTT/tasks/02_tables_figures_captions.md`
  - `WE-FTT/tasks/03_manuscript_revision_3nd.md`

## 建议目录结构（后续按任务逐步落地）

```
evaluation_protocol/
  leakage_audit/        # 权重/聚类/关联规则泄漏审计证据
  data_splits/          # 事件级切分清单（版本化）
  placebo/              # Placebo 生成 + 置换检验
  era5/                 # ERA5 下载/对齐 + 海洋条件化
  noise_budget/         # 噪声预算与 σ_eff 计算
  land_conditioning/    # 陆地协变量残差化/条件化
  run_evaluation.py     # 一键评估入口（聚合调用）
  reproduce_results.sh  # 一键复现实验（封装命令）
```

> 注意：论文最终引用用的图/表需要拷贝到 `WE-FTT/docs/revision_codex/revisions-3nd/`，避免 LaTeX 依赖跨目录路径。

