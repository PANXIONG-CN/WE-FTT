# 修订任务 04：T2 事件级严格切分与区域外推验证

> 目标：把 `comprehensive-analysis-and-execution-plan.md` 的 T2 拆成可执行清单，并在仓库内形成“已完成/待完成”状态。
>
> 参考：`WE-FTT/docs/revision_codex/ai-reviews/comprehensive-analysis-and-execution-plan.md`（T2）

## 0. 范围定义（T2 原始要求）

- 事件级切分（同一 `event_id` 不跨 train/val/test）
- 10°×10° 区域留一（spatial leave-one-block-out）外推验证
- 形成可审计脚本 + 结果表/图 + 文稿引用

## 1. 当前已完成

- [x] 事件级切分清单：`WE-FTT/evaluation_protocol/data_splits/event_grouped_splits_v1.json`
- [x] 切分脚本：`WE-FTT/evaluation_protocol/data_splits/make_event_splits.py`
- [x] 切分文档：`WE-FTT/evaluation_protocol/data_splits/README.md`
- [x] 正文已写入“事件级切分 + 防泄漏”描述：`WE-FTT/docs/revision_codex/revisions-3nd/main_clean.tex`

## 2. 原待完成项（已闭环）

- [x] 区域留一脚本：`WE-FTT/scripts/run_spatial_cv.py`（或协议目录下等价脚本）
- [x] 10°×10° 留一结果产物（CSV/JSON + 图）
  - `WE-FTT/evaluation_protocol/data_splits/tables/tab_s11_spatial_cv_blocks.csv`
  - `WE-FTT/evaluation_protocol/data_splits/results/spatial_cv_summary_v1.json`
  - `WE-FTT/evaluation_protocol/data_splits/figures/fig_s8_spatial_cv_mcc.png`
  - 结果摘要：`n_blocks_evaluated=27`，`weighted_mean_mcc=-0.0073`
  - 改进版对比（v2 诊断）：
    - 基线复跑：`WE-FTT/evaluation_protocol/data_splits/results/spatial_cv_summary_v2_baseline.json`（`weighted_mean_mcc=-0.0086`）
    - 改进配置（HGB + day_mean + FPR 约束阈值）：`WE-FTT/evaluation_protocol/data_splits/results/spatial_cv_summary_v2_hgb_daymean.json`（`weighted_mean_mcc=0.0050`，`weighted_mean_fpr=0.0521`）
    - 改进配置（HGB + day_mean + zone-wise）：`WE-FTT/evaluation_protocol/data_splits/results/spatial_cv_summary_v2_hgb_daymean_zonewise.json`（`weighted_mean_mcc=-0.0056`）
  - 改进版对比（v3：区域自适应阈值 + 条件化特征）：
    - `WE-FTT/evaluation_protocol/data_splits/results/spatial_cv_summary_v3_hgb_daymean_zoneThr.json`（`weighted_mean_mcc=0.0082`，`weighted_mean_fpr=0.0379`）
    - `WE-FTT/evaluation_protocol/data_splits/results/spatial_cv_summary_v3_hgb_pixel_landResid_zoneThr.json`（`weighted_mean_mcc=0.0124`，`weighted_mean_fpr=0.1252`）
    - 对应脚本增强：`WE-FTT/scripts/run_spatial_cv.py` 已支持 `--zone_adaptive_thresholds` 与 `--land_residualize`（含 ERA5/geo-season/zone-wise 背景残差化）
- [x] 在文稿中补充区域外推结果与失败区域说明（若有）
  - `WE-FTT/docs/revision_codex/revisions-3nd/main_clean.tex`

## 3. 验收标准（DoD）

- [x] 事件级切分可复现（固定随机种子、清单可版本化）
- [x] 区域留一评估可复现（脚本 + 参数 + 结果文件）
- [x] 区域外推性能结论进入主文或补充材料

## 4. 结论（当前状态）

- **T2 当前状态：已完成（含事件级切分与区域留一）**
- **v3 结论补充**：删除“空间外推能力显著提升”的预期表述。当前证据显示在严格空间外推下 MCC 仍接近 0（最佳配置 `weighted_mean_mcc=0.0124`），但 FPR 可被稳定控制；该结果支持“区域迁移能力有限、但评估协议与风险边界可复现”的叙事。
