#!/usr/bin/env python3
"""
T6 ERA5 下载/对齐（v1，最小可用实现：对齐 + 缓存）。

现阶段约束：
- 评估协议优先保证“对齐与可重复运行”；下载部分依赖 CDS API 凭据，默认不自动下载。

输入（假设用户已下载 ERA5 hourly 单层数据到本地）：
- `--era5_hourly_nc`：一个或多个 NetCDF 文件（支持 glob），包含变量：
  - 10m u wind: u10
  - 10m v wind: v10
  - total precipitation: tp

输出（缓存到 evaluation_protocol/era5/cache/daily/）：
- `wind_speed_YYYYMMDD.npy`（m/s，日均）
- `precip_mm_YYYYMMDD.npy`（mm/day，日累计）

对齐策略（KISS）：
- 目标网格采用 AMSR2-L3 0.25° (720×1440) 的 i/j 定义（见 common/geo.py）
- 使用 xarray 的 nearest 插值到目标 lat/lon（避免复杂插值引入额外误差）
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np

# 将 WE-FTT 根目录加入路径（保证可直接以脚本方式运行）
weftt_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if weftt_root not in sys.path:
    sys.path.insert(0, weftt_root)

from evaluation_protocol.common.geo import GRID_N_LAT, GRID_N_LON, grid_lat, grid_lon  # noqa: E402
from evaluation_protocol.common.jsonl import write_json  # noqa: E402
from evaluation_protocol.common.logging_utils import setup_logging  # noqa: E402
from evaluation_protocol.common.paths import get_repo_paths, resolve_path  # noqa: E402


logger = logging.getLogger(__name__)


def _require_xarray():
    try:
        import xarray as xr  # noqa: F401
    except Exception as e:
        raise ImportError("缺少依赖 xarray/netCDF4。请先安装后重试。") from e


def _target_lat_lon():
    lats = np.array([grid_lat(i) for i in range(GRID_N_LAT)], dtype=np.float32)
    lons = np.array([grid_lon(j) for j in range(GRID_N_LON)], dtype=np.float32)
    return lats, lons


def _normalize_lon_180(lon: np.ndarray) -> np.ndarray:
    x = (lon + 180.0) % 360.0 - 180.0
    x[x == 180.0] = -180.0
    return x


def _detect_coord_names(ds) -> tuple[str, str, str]:
    # ERA5 常见：time, latitude, longitude
    for t in ("time", "valid_time"):
        if t in ds.coords:
            time_name = t
            break
    else:
        raise ValueError("未找到 time 坐标（期望 time/valid_time）。")

    lat_name = "latitude" if "latitude" in ds.coords else ("lat" if "lat" in ds.coords else None)
    lon_name = "longitude" if "longitude" in ds.coords else ("lon" if "lon" in ds.coords else None)
    if lat_name is None or lon_name is None:
        raise ValueError("未找到 latitude/longitude 坐标（期望 latitude/longitude 或 lat/lon）。")
    return time_name, lat_name, lon_name


def align_daily_cache(
    *,
    era5_nc_paths: Sequence[Path],
    out_daily_dir: Path,
    start: Optional[str],
    end: Optional[str],
    overwrite: bool,
) -> dict:
    _require_xarray()
    import xarray as xr

    out_daily_dir.mkdir(parents=True, exist_ok=True)
    lats_t, lons_t = _target_lat_lon()

    logger.info("打开 ERA5 NetCDF（n=%d）", len(era5_nc_paths))
    # xarray.open_mfdataset 默认依赖 dask；为保持最小依赖（KISS），这里显式用 open_dataset + merge。
    opened = [xr.open_dataset(str(p)) for p in era5_nc_paths]
    try:
        ds = opened[0] if len(opened) == 1 else xr.merge(opened, compat="no_conflicts", join="outer")
        time_name, lat_name, lon_name = _detect_coord_names(ds)

        # 统一经度到 [-180,180)
        lon_vals = ds[lon_name].values
        if np.nanmax(lon_vals) > 180.0:
            ds = ds.assign_coords({lon_name: _normalize_lon_180(ds[lon_name].values)})
            ds = ds.sortby(lon_name)

        if start or end:
            ds = ds.sel({time_name: slice(start, end)})

        need_vars = ["u10", "v10", "tp"]
        for v in need_vars:
            if v not in ds.data_vars:
                raise ValueError(f"ERA5 输入缺少变量: {v}")

        # wind speed（m/s）日均
        ws = np.sqrt(ds["u10"] ** 2 + ds["v10"] ** 2)
        ws_daily = ws.resample({time_name: "1D"}).mean()

        # precipitation（m -> mm）日累计
        tp_mm = ds["tp"] * 1000.0
        tp_daily = tp_mm.resample({time_name: "1D"}).sum()

        days = ws_daily[time_name].dt.date.values
        written = 0
        skipped = 0

        for idx, d in enumerate(days):
            d_py = date.fromisoformat(str(d))
            ymd = d_py.strftime("%Y%m%d")
            out_ws = out_daily_dir / f"wind_speed_{ymd}.npy"
            out_tp = out_daily_dir / f"precip_mm_{ymd}.npy"
            if (out_ws.exists() and out_tp.exists()) and not overwrite:
                skipped += 1
                continue

            ws_day = ws_daily.isel({time_name: idx})
            tp_day = tp_daily.isel({time_name: idx})

            # nearest 对齐到目标网格
            ws_aligned = ws_day.interp({lat_name: lats_t, lon_name: lons_t}, method="nearest")
            tp_aligned = tp_day.interp({lat_name: lats_t, lon_name: lons_t}, method="nearest")

            ws_arr = np.asarray(ws_aligned.values, dtype=np.float32)
            tp_arr = np.asarray(tp_aligned.values, dtype=np.float32)
            if ws_arr.shape != (GRID_N_LAT, GRID_N_LON) or tp_arr.shape != (GRID_N_LAT, GRID_N_LON):
                raise RuntimeError(f"对齐后 shape 异常：ws={ws_arr.shape} tp={tp_arr.shape}")

            np.save(out_ws, ws_arr)
            np.save(out_tp, tp_arr)
            written += 1

        return {"written_days": int(written), "skipped_days": int(skipped), "total_days": int(len(days))}
    finally:
        for ds0 in opened:
            try:
                ds0.close()
            except Exception:
                pass


def parse_args():
    repo = get_repo_paths()
    default_out_dir = repo.eval_root / "era5" / "cache" / "daily"
    default_meta = repo.eval_root / "era5" / "results" / "era5_daily_cache_v1.meta.json"

    p = argparse.ArgumentParser(description="Align ERA5 hourly to AMSR2 grid and cache daily arrays")
    p.add_argument("--era5_hourly_nc", type=str, required=True, help="ERA5 hourly NetCDF 路径或glob（包含u10/v10/tp）")
    p.add_argument("--out_daily_dir", type=str, default=str(default_out_dir))
    p.add_argument("--start", type=str, default=None, help="可选：起始日期（YYYY-MM-DD）")
    p.add_argument("--end", type=str, default=None, help="可选：结束日期（YYYY-MM-DD）")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--out_meta_json", type=str, default=str(default_meta))
    p.add_argument("--log_file", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging(log_file=args.log_file)

    nc_glob = str(args.era5_hourly_nc)
    nc_paths = [Path(p) for p in glob.glob(nc_glob)]
    if not nc_paths:
        raise FileNotFoundError(f"未匹配到 ERA5 NetCDF: {nc_glob}")

    out_daily_dir = resolve_path(args.out_daily_dir)
    out_meta = resolve_path(args.out_meta_json)
    out_meta.parent.mkdir(parents=True, exist_ok=True)

    stats = align_daily_cache(
        era5_nc_paths=nc_paths,
        out_daily_dir=out_daily_dir,
        start=args.start,
        end=args.end,
        overwrite=bool(args.overwrite),
    )

    write_json(
        out_meta,
        {
            "version": "era5_daily_cache_v1",
            "generated_at_utc": datetime.utcnow().isoformat() + "Z",
            "era5_hourly_nc": nc_glob,
            "out_daily_dir": str(out_daily_dir),
            "start": args.start,
            "end": args.end,
            "overwrite": bool(args.overwrite),
            "stats": stats,
        },
    )
    logger.info("写入 meta: %s", out_meta)


if __name__ == "__main__":
    main()
