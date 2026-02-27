#!/usr/bin/env python3
"""
使用 CDS API 下载 ERA5 hourly single levels，并对齐到 AMSR2 网格生成日缓存（v1）。

为什么要单独做一个下载脚本？
- ERA5 下载需要用户配置 CDS 凭据（~/.cdsapirc），且下载量较大，适合与对齐/评估解耦。

默认策略（KISS）：
- 以 eval_parquet（Zone A / type=0）的 `sample_date` 作为“需要的日期集合”
- 按 (year, month) 聚合下载，每月一个 NetCDF
- 下载后调用 `download_and_align.align_daily_cache()` 写入 daily npy 缓存

前置：
- 安装依赖：`pip install cdsapi xarray netCDF4`
- 配置 `~/.cdsapirc`（包含 url/key），并在 CDS 页面接受 ERA5 许可条款
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import zipfile
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd

# 将 WE-FTT 根目录加入路径（保证可直接以脚本方式运行）
weftt_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if weftt_root not in sys.path:
    sys.path.insert(0, weftt_root)

from evaluation_protocol.common.jsonl import write_json  # noqa: E402
from evaluation_protocol.common.logging_utils import setup_logging  # noqa: E402
from evaluation_protocol.common.paths import get_repo_paths, resolve_path  # noqa: E402
from evaluation_protocol.era5.download_and_align import align_daily_cache  # noqa: E402
from evaluation_protocol.common.splits import load_event_splits  # noqa: E402


logger = logging.getLogger(__name__)


def _require_cdsapi():
    try:
        import cdsapi  # noqa: F401
    except Exception as e:
        raise ImportError("缺少依赖 cdsapi。请先在 venv 中安装：pip install cdsapi") from e


def _infer_needed_days(
    eval_parquet: Path,
    *,
    zone_type: int,
    splits_json: Path,
    use_split: str,
    overwrite: bool,
    out_daily_dir: Path,
) -> List[date]:
    cols = ["sample_date", "zone_type", "event_id"]
    df = pd.read_parquet(eval_parquet, columns=cols)
    df = df[df["zone_type"].astype(int) == int(zone_type)].copy()
    if df.empty:
        raise ValueError(f"eval_parquet 中未找到 zone_type={zone_type} 的样本，无法推断下载日期。")

    use_split = str(use_split).strip().lower()
    if use_split != "all":
        splits = load_event_splits(splits_json)
        if use_split == "train":
            allow = splits.train_event_ids
        elif use_split == "val":
            allow = splits.val_event_ids
        elif use_split == "test":
            allow = splits.test_event_ids
        else:
            raise ValueError("--use_split 必须为 train/val/test/all")
        df = df[df["event_id"].astype(str).isin(allow)].copy()
        if df.empty:
            raise ValueError(f"按 use_split={use_split} 过滤后样本为空，无法推断下载日期。")

    ds = pd.to_datetime(df["sample_date"], errors="coerce").dt.date.dropna().unique().tolist()
    days = sorted([date.fromisoformat(str(x)) for x in ds])

    if overwrite:
        return days

    keep: List[date] = []
    for d in days:
        ymd = d.strftime("%Y%m%d")
        ws = out_daily_dir / f"wind_speed_{ymd}.npy"
        pr = out_daily_dir / f"precip_mm_{ymd}.npy"
        if ws.exists() and pr.exists():
            continue
        keep.append(d)
    return keep


def _group_by_year_month(days: Sequence[date]) -> Dict[Tuple[int, int], List[date]]:
    out: Dict[Tuple[int, int], List[date]] = {}
    for d in days:
        out.setdefault((d.year, d.month), []).append(d)
    return out


def _cds_request_for_month(year: int, month: int, days: Sequence[date], *, area: Sequence[float]):
    day_list = [f"{d.day:02d}" for d in sorted(days)]
    time_list = [f"{h:02d}:00" for h in range(24)]
    return {
        "product_type": "reanalysis",
        "format": "netcdf",
        "variable": [
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "total_precipitation",
        ],
        "year": f"{int(year):04d}",
        "month": f"{int(month):02d}",
        "day": day_list,
        "time": time_list,
        "grid": [0.25, 0.25],
        "area": [float(x) for x in area],  # [N, W, S, E]
    }


def _extract_zip_to_nc_paths(zip_path: Path, *, out_dir: Path) -> List[Path]:
    """
    CDS 有时会返回 zip（内部包含一个或多个 .nc）。
    这里将 zip 解包为独立的 .nc 文件并返回路径列表。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_paths: List[Path] = []
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            name = Path(info.filename).name
            if not name:
                continue
            if name.endswith("instant.nc"):
                out_name = f"{zip_path.stem}_instant.nc"
            elif name.endswith("accum.nc"):
                out_name = f"{zip_path.stem}_accum.nc"
            else:
                out_name = f"{zip_path.stem}_{name}"
            out_fp = out_dir / out_name

            # 若已存在且大小一致，直接复用（支持断点/重跑）
            try:
                if out_fp.exists() and out_fp.stat().st_size == int(info.file_size):
                    out_paths.append(out_fp)
                    continue
            except Exception:
                pass

            with zf.open(info, "r") as src, open(out_fp, "wb") as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)
            out_paths.append(out_fp)
    if not out_paths:
        raise RuntimeError(f"zip 内未找到可用的 NetCDF 文件：{zip_path}")
    return out_paths


def _prepare_era5_nc_paths(target_path: Path, *, out_nc_dir: Path) -> Tuple[List[Path], bool]:
    """
    返回用于对齐的 NetCDF 路径列表，以及 target_path 是否为 zip。
    """
    try:
        is_zip = zipfile.is_zipfile(target_path)
    except Exception:
        is_zip = False
    if not is_zip:
        return [target_path], False
    nc_paths = _extract_zip_to_nc_paths(target_path, out_dir=out_nc_dir)
    return nc_paths, True


def download_and_cache(
    *,
    eval_parquet: Path,
    out_nc_dir: Path,
    out_daily_dir: Path,
    zone_type: int,
    splits_json: Path,
    use_split: str,
    overwrite: bool,
    cleanup_nc: bool,
    dry_run: bool,
    area: Sequence[float],
) -> dict:
    if not dry_run:
        _require_cdsapi()
        import cdsapi  # noqa: F401

    out_nc_dir.mkdir(parents=True, exist_ok=True)
    out_daily_dir.mkdir(parents=True, exist_ok=True)

    need_days = _infer_needed_days(
        eval_parquet,
        zone_type=int(zone_type),
        splits_json=splits_json,
        use_split=str(use_split),
        overwrite=bool(overwrite),
        out_daily_dir=out_daily_dir,
    )
    if not need_days:
        logger.info("daily cache 已齐全（无需下载）。")
        return {"months": 0, "days": 0, "downloads": []}

    by_month = _group_by_year_month(need_days)
    logger.info("需要下载的日期数：%d（月数=%d）", len(need_days), len(by_month))

    downloads = []
    client = None
    if not dry_run:
        import cdsapi

        client = cdsapi.Client()

    for (y, m), days in sorted(by_month.items()):
        target = out_nc_dir / f"era5_hourly_{y:04d}{m:02d}.nc"
        req = _cds_request_for_month(y, m, days, area=area)
        downloads.append({"year": y, "month": m, "days": [d.isoformat() for d in days], "target_nc": str(target)})

        if dry_run:
            logger.info("[dry-run] %04d-%02d days=%d -> %s", y, m, len(days), target)
            continue

        if target.exists():
            logger.info("已存在，跳过下载：%s", target)
        else:
            logger.info("下载 ERA5：%04d-%02d days=%d", y, m, len(days))
            assert client is not None
            client.retrieve("reanalysis-era5-single-levels", req, str(target))

        nc_paths, target_is_zip = _prepare_era5_nc_paths(target, out_nc_dir=out_nc_dir)

        # 对齐并写 daily cache（只写缺失的日文件；overwrite 由 align_daily_cache 控制）
        align_daily_cache(
            era5_nc_paths=nc_paths,
            out_daily_dir=out_daily_dir,
            start=None,
            end=None,
            overwrite=bool(overwrite),
        )

        if cleanup_nc:
            # 先清理解包后的 NetCDF
            for p in nc_paths:
                try:
                    p.unlink()
                    logger.info("清理临时 NetCDF：%s", p)
                except Exception as e:
                    logger.warning("清理 NetCDF 失败（忽略）：%s（%s）", p, e)
            # 若原始下载是 zip，也一并清理
            if target_is_zip:
                try:
                    target.unlink()
                    logger.info("清理临时 zip：%s", target)
                except Exception as e:
                    logger.warning("清理 zip 失败（忽略）：%s（%s）", target, e)

    return {"months": int(len(by_month)), "days": int(len(need_days)), "downloads": downloads}


def parse_args():
    repo = get_repo_paths()
    default_eval = repo.eval_root / "datasets" / "mbt_eval_samples_v1.parquet"
    default_splits = repo.eval_root / "data_splits" / "event_grouped_splits_v1.json"
    default_nc = repo.eval_root / "era5" / "cache" / "hourly_nc"
    default_daily = repo.eval_root / "era5" / "cache" / "daily"
    default_meta = repo.eval_root / "era5" / "results" / "era5_download_v1.meta.json"

    p = argparse.ArgumentParser(description="Download ERA5 via CDS and build daily cache (v1)")
    p.add_argument("--eval_parquet", type=str, default=str(default_eval))
    p.add_argument("--splits_json", type=str, default=str(default_splits))
    p.add_argument("--use_split", type=str, default="test", choices=["train", "val", "test", "all"], help="仅下载指定 split 覆盖的日期（默认 test）")
    p.add_argument("--zone_type", type=int, default=0, help="仅为指定分区下载（T6默认海洋=0）")
    p.add_argument("--out_nc_dir", type=str, default=str(default_nc))
    p.add_argument("--out_daily_dir", type=str, default=str(default_daily))
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--cleanup_nc", action="store_true", help="生成 daily cache 后删除每月 NetCDF（节省空间）")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--area", type=str, default="70,-180,-70,180", help="下载区域 bbox：N,W,S,E（默认与lat_limit一致）")
    p.add_argument("--out_meta_json", type=str, default=str(default_meta))
    p.add_argument("--log_file", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging(log_file=args.log_file)

    eval_parquet = resolve_path(args.eval_parquet)
    splits_json = resolve_path(args.splits_json)
    out_nc_dir = resolve_path(args.out_nc_dir)
    out_daily_dir = resolve_path(args.out_daily_dir)
    out_meta = resolve_path(args.out_meta_json)
    out_meta.parent.mkdir(parents=True, exist_ok=True)

    area = [float(x) for x in str(args.area).split(",")]
    if len(area) != 4:
        raise ValueError("--area 需要 4 个数：N,W,S,E")

    stats = download_and_cache(
        eval_parquet=eval_parquet,
        out_nc_dir=out_nc_dir,
        out_daily_dir=out_daily_dir,
        zone_type=int(args.zone_type),
        splits_json=splits_json,
        use_split=str(args.use_split),
        overwrite=bool(args.overwrite),
        cleanup_nc=bool(args.cleanup_nc),
        dry_run=bool(args.dry_run),
        area=area,
    )

    write_json(
        out_meta,
        {
            "version": "era5_download_v1",
            "generated_at_utc": datetime.utcnow().isoformat() + "Z",
            "eval_parquet": str(eval_parquet),
            "splits_json": str(splits_json),
            "use_split": str(args.use_split),
            "zone_type": int(args.zone_type),
            "out_nc_dir": str(out_nc_dir),
            "out_daily_dir": str(out_daily_dir),
            "overwrite": bool(args.overwrite),
            "cleanup_nc": bool(args.cleanup_nc),
            "dry_run": bool(args.dry_run),
            "area": area,
            "stats": stats,
        },
    )
    logger.info("写入 meta: %s", out_meta)


if __name__ == "__main__":
    main()
