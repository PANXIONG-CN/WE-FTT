# data_splits（事件级切分清单｜版本化）

本目录用于承载 **事件级（event-grouped）** 的 train/val/test 切分清单，确保：
- 同一地震事件样本不会跨集合（防止泄漏）
- 切分可复现（固定随机种子）
- 可审计（记录生成条件与版本）

## 事件 ID 定义

- 默认采用 `WE-FTT/data/raw/earthquake_catalog.csv` 的 `id` 字段（USGS event id）。
- 若某行缺失 `id`，会生成稳定哈希 ID（以 date/time/lat/lon/mag 等拼接后 sha1）。

## 当前版本

- `event_grouped_splits_v1.json`
  - 过滤条件：`mag>=7.0`；且海洋区（type=0）仅取 `depth<70km`
  - 分层：按 `Type_`（0..4）分层切分，尽量保证各区在 val/test 仍有代表

## 生成脚本

```bash
python WE-FTT/evaluation_protocol/data_splits/make_event_splits.py \
  --catalog_csv "WE-FTT/data/raw/earthquake_catalog.csv" \
  --out_path "WE-FTT/evaluation_protocol/data_splits/event_grouped_splits_v1.json" \
  --seed 42
```

