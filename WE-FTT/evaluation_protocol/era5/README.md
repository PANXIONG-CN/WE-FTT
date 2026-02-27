# era5（T6 海洋区条件化）

本目录包含两类脚本：

- **下载/对齐/缓存**：把 ERA5 的风场/降水按日聚合，并对齐到 AMSR2-L3 0.25° 网格，写入 `.npy` 日缓存
- **条件化评估**：在 Zone A（type=0）对 test 集做硬掩膜（风速/降水阈值），比较 FPR 前后变化（含 CI）

## 1) 配置 CDS API（下载所需）

1. 申请/登录 Copernicus CDS 账号，并接受 ERA5 数据集条款
2. 配置 `~/.cdsapirc`（示例，仅示意格式）

```txt
url: https://cds.climate.copernicus.eu/api
key: <uid>:<api-key>
```

## 2) 下载 ERA5 并生成 daily cache（推荐）

默认会从 `eval_parquet` 推断 **Zone A 的 sample_date 集合**，按月下载并生成：

- `wind_speed_YYYYMMDD.npy`（m/s，日均）
- `precip_mm_YYYYMMDD.npy`（mm/day，日累计）

```bash
PY="WE-FTT/.venv/bin/python"
EVAL="WE-FTT/evaluation_protocol/datasets/mbt_eval_samples_v1.parquet"

"${PY}" "WE-FTT/evaluation_protocol/era5/download_from_cds.py" \
  --eval_parquet "${EVAL}" \
  --use_split test \
  --zone_type 0 \
  --area "70,-180,-70,180" \
  --cleanup_nc
```

> `--cleanup_nc` 会删除每月 NetCDF（仅保留 `.npy` 日缓存），避免磁盘占用过大。

## 3) 运行海洋硬掩膜条件化（T6 路线1）

```bash
PY="WE-FTT/.venv/bin/python"
EVAL="WE-FTT/evaluation_protocol/datasets/mbt_eval_samples_v1.parquet"
CACHE="WE-FTT/evaluation_protocol/era5/cache/daily"

"${PY}" "WE-FTT/evaluation_protocol/era5/ocean_conditioning.py" \
  --eval_parquet "${EVAL}" \
  --era5_daily_cache_dir "${CACHE}" \
  --wind_thresh 10 \
  --precip_thresh 5 \
  --use_weights
```

产物默认写入：

- `WE-FTT/evaluation_protocol/era5/tables/tab_s7_era5_ocean_mask_v1.csv`
- `WE-FTT/evaluation_protocol/era5/figures/fig_s6_era5_ocean_fpr_v1.png`
- `WE-FTT/evaluation_protocol/era5/results/ocean_conditioning_v1.json`

## 3.1)（推荐）v2：以“抑制报警”做条件化（避免改变分母）

v1 的“掩膜”实现为 **删除高风/高降水样本** 后重算 FPR；这会改变负类分母。

v2 更贴近 operational screening：当 ERA5 超阈值时 **强制该样本预测为负类（不报警）**，
因此 FPR 的下降来自“屏蔽掉的假阳性报警”，而非样本选择。

此外，v2 会在 val 上做两级校准：

- 选择分类阈值，使 baseline FPR 接近 0.242（用于对齐论文 Zone A baseline 量级）
- 在候选风/降水阈值网格中选择满足 conditioned FPR <= 0.10 且 MCC 最大的组合

```bash
PY="WE-FTT/.venv/bin/python"
EVAL="WE-FTT/evaluation_protocol/datasets/mbt_eval_samples_v1.parquet"
CACHE="WE-FTT/evaluation_protocol/era5/cache/daily"

"${PY}" "WE-FTT/evaluation_protocol/era5/ocean_conditioning_v2.py" \
  --eval_parquet "${EVAL}" \
  --era5_daily_cache_dir "${CACHE}" \
  --use_weights
```

产物默认写入：

- `WE-FTT/evaluation_protocol/era5/tables/tab_s7_era5_ocean_mask_v2.csv`
- `WE-FTT/evaluation_protocol/era5/figures/fig_s6_era5_ocean_fpr_v2.png`
- `WE-FTT/evaluation_protocol/era5/results/ocean_conditioning_v2.json`

## 4) 运行海洋 ERA5 残差化条件化（T6 路线2）

> 残差化需要使用 **训练折内的对照窗样本** 拟合背景模型，因此必须保证 ERA5 daily cache 覆盖 `--use_split all` 的日期集合。

### 4.1 补齐 Zone A 所有 split 的 ERA5 daily cache（推荐）

```bash
PY="WE-FTT/.venv/bin/python"
EVAL="WE-FTT/evaluation_protocol/datasets/mbt_eval_samples_v1.parquet"

"${PY}" "WE-FTT/evaluation_protocol/era5/download_from_cds.py" \
  --eval_parquet "${EVAL}" \
  --use_split all \
  --zone_type 0 \
  --area "70,-180,-70,180" \
  --cleanup_nc
```

### 4.2 残差化评估（v1）

```bash
PY="WE-FTT/.venv/bin/python"
EVAL="WE-FTT/evaluation_protocol/datasets/mbt_eval_samples_v1.parquet"
CACHE="WE-FTT/evaluation_protocol/era5/cache/daily"

"${PY}" "WE-FTT/evaluation_protocol/era5/ocean_residualize.py" \
  --eval_parquet "${EVAL}" \
  --era5_daily_cache_dir "${CACHE}" \
  --alpha 1.0 \
  --use_weights
```

产物默认写入：

- `WE-FTT/evaluation_protocol/era5/tables/tab_s7_era5_ocean_residualize_v1.csv`
- `WE-FTT/evaluation_protocol/era5/figures/fig_s6_era5_ocean_residualize_v1.png`
- `WE-FTT/evaluation_protocol/era5/results/ocean_residualize_v1.json`
