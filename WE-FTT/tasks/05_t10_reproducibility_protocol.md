# 修订任务 05：T10 公开可复现评估协议

> 目标：对齐 `comprehensive-analysis-and-execution-plan.md` 的 T10，形成“协议脚本齐全 + 对外入口清晰”的落地状态。
>
> 参考：`WE-FTT/docs/revision_codex/ai-reviews/comprehensive-analysis-and-execution-plan.md`（T10）

## 0. 范围定义（T10 原始要求）

- `evaluation_protocol/README.md`
- `evaluation_protocol/data_splits/`（事件级切分清单）
- `evaluation_protocol/placebo/generate_placebos.py`
- `evaluation_protocol/era5/download_and_align.py`
- `evaluation_protocol/run_evaluation.py`
- `evaluation_protocol/reproduce_results.sh`
- GitHub README 增加 `Reproducibility` 章节

## 1. 当前已完成

- [x] 协议目录与说明：`WE-FTT/evaluation_protocol/README.md`
- [x] 事件级切分目录：`WE-FTT/evaluation_protocol/data_splits/`
- [x] Placebo 生成脚本：`WE-FTT/evaluation_protocol/placebo/generate_placebos.py`
- [x] ERA5 对齐脚本：`WE-FTT/evaluation_protocol/era5/download_and_align.py`
- [x] 一键评估入口：`WE-FTT/evaluation_protocol/run_evaluation.py`
- [x] 一键复现脚本：`WE-FTT/evaluation_protocol/reproduce_results.sh`

## 2. 原待完成项（已闭环）

- [x] `WE-FTT/README.md` 增加 `Reproducibility` 章节并链接 `evaluation_protocol/`

## 3. 验收标准（DoD）

- [x] 评估协议核心脚本与目录齐全
- [x] 协议入口文档可独立指导复现
- [x] 项目根 README 暴露可复现入口（Reproducibility）

## 4. 结论（当前状态）

- **T10 当前状态：已完成（协议落地 + README 对外入口）**
