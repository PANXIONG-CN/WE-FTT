# datasets（评估数据集｜带元数据）

本目录用于从 **原始 AMSR2-L3 HDF5 网格**重建“带事件/时空元数据”的评估样本，作为后续：
- 事件级切分（T2）
- 时空 placebo（T4）
- ERA5 条件化（T6）
- 陆地残差化（T7）

的共同数据基础。

## 约定（已锁定）

- 网格：EQR 0.25°，shape = (720, 1440)
- 轨道：Descending（夜间）`EQMD`
- 均值：`01D`（DayMean）
- 特征：10 通道（06/10/23/36/89 GHz × H/V）
- 事件目录：`WE-FTT/data/raw/earthquake_catalog.csv`
- 海洋区（Zone A / type=0）仅取浅源（depth < 70 km）

## 生成脚本

- `build_mbt_eval_dataset.py`：生成 `mbt_eval_samples_v1.parquet`

示例：

```bash
python WE-FTT/evaluation_protocol/datasets/build_mbt_eval_dataset.py \
  --amsr2_root "/mnt/hdd_4tb_data/ArchivedWorks/MBTpaper_raw_data/AMSR2" \
  --out_path "WE-FTT/evaluation_protocol/datasets/mbt_eval_samples_v1.parquet" \
  --min_mag 7.0 \
  --pre_days 20 --post_days 10 \
  --pixels_per_event_day 200 \
  --control_dates_per_event 2 \
  --doy_window 15 \
  --seed 42
```

> 依赖：`numpy`、`h5py`、`pyarrow`（用于 Parquet）。若未安装，脚本会给出提示。

### 输出说明（v1）

- `flag=1`：真实事件窗（anchor_date = event_date）
- `flag=0`：对照窗（anchor_date 为采样得到的“伪事件日期”，保证不与任何真实事件窗重叠，且控制同季节 DOY）
