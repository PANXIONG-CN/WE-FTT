补充实验 #3：海啸与极端风浪伪影的敏感性分析（Marine Zone Controls）

目标：验证海域检出不是由海啸或极端风浪伪影造成的。在“主震前窗口＋海啸到达±48h排除＋极端风浪日剔除”后，检出是否仍显著高于随机。

目录结构：

exp3/
- scripts/
  - exp3_common.py              # 公共函数：事件筛选、海啸检索、风浪代理、控制与统计
  - generate_fig_s3.py          # 生成 Figure S3 条形图与题注
  - generate_table_s3.py        # 生成 Table S3 指标表
- data/
  - earthquake_candidates.csv   # 从 data/raw/earthquake_catalog.csv 筛选的海域候选（M≥7.0）
  - tsunami_events.json         # NOAA/NCEI hazard-service 检索缓存
- figures/
  - fig_s3_marine_controls.(pdf|png|svg)
- tables/
  - table_s3_marine_controls.md
- docs/
  - FIG_S3_CAPTION.md

使用说明：
- 运行 `python supplement_experiments/exp3/scripts/generate_fig_s3.py` 生成图件与题注。
- 运行 `python supplement_experiments/exp3/scripts/generate_table_s3.py` 生成表格。

依赖：复用 `supplement_experiments/utils.py` 与 `supplement_experiments/nature_style.py`，网络可用时调用 NOAA/NCEI hazard-service。

