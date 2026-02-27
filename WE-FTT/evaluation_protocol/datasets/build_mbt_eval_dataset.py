#!/usr/bin/env python3
"""
从原始 AMSR2-L3 HDF5 网格重建“带元数据”的评估样本集（v1）。

设计目标：
- 可审计：每条样本包含 event_id、sample_date、day_offset、像元坐标等
- 可复现：固定 seed；输出 schema 固定
- KISS：先实现真实事件窗（flag=1）；对照/placebo 由后续脚本按需生成

输出（默认）：
WE-FTT/evaluation_protocol/datasets/mbt_eval_samples_v1.parquet
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import asdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# 将 WE-FTT 根目录加入路径（保证可直接以脚本方式运行）
weftt_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if weftt_root not in sys.path:
    sys.path.insert(0, weftt_root)

from evaluation_protocol.common.amsr2 import AMSR2DailyGrid, AMSR2Spec, FEATURE_COLUMNS  # noqa: E402
from evaluation_protocol.common.catalog import EarthquakeEvent, load_events  # noqa: E402
from evaluation_protocol.common.geo import dobrovolsky_radius_km  # noqa: E402
from evaluation_protocol.common.jsonl import write_json  # noqa: E402
from evaluation_protocol.common.logging_utils import setup_logging  # noqa: E402
from evaluation_protocol.common.paths import get_repo_paths, resolve_path  # noqa: E402
from evaluation_protocol.common.sampling import (  # noqa: E402
    build_excluded_dates,
    sample_control_dates,
    sample_points_within_radius,
)


logger = logging.getLogger(__name__)


def _require_pyarrow():
    try:
        import pyarrow as pa  # noqa: F401
        import pyarrow.parquet as pq  # noqa: F401
    except Exception as e:
        raise ImportError(
            "缺少依赖 pyarrow，无法写入 Parquet。请在你的运行环境中安装后重试。"
        ) from e


def _iter_window_days(event_date: date, pre_days: int, post_days: int):
    for d in range(-int(pre_days), int(post_days) + 1):
        yield d, (event_date + timedelta(days=d))


def _write_parquet_stream(out_path: Path, rows_iter, *, batch_size: int = 200_000) -> Dict[str, int]:
    _require_pyarrow()
    import pyarrow as pa
    import pyarrow.parquet as pq

    out_path.parent.mkdir(parents=True, exist_ok=True)

    writer: Optional[pq.ParquetWriter] = None
    total = 0
    kept = 0
    dropped_nan = 0
    buf: Dict[str, List] = {}

    def flush():
        nonlocal writer, total, kept, dropped_nan, buf
        if not buf:
            return
        table = pa.table(buf)
        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema, compression="zstd")
        writer.write_table(table)
        buf = {}

    for row in rows_iter:
        total += 1
        # row 是 dict（含 np 标量）；pyarrow 能处理，但我们做一次轻量校验：特征 NaN 直接丢弃
        has_nan = False
        for k in FEATURE_COLUMNS:
            v = row.get(k, None)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                has_nan = True
                break
        if has_nan:
            dropped_nan += 1
            continue

        kept += 1
        for k, v in row.items():
            buf.setdefault(k, []).append(v)
        if kept % batch_size == 0:
            flush()

    flush()
    if writer is not None:
        writer.close()
    return {"total_rows": total, "kept_rows": kept, "dropped_nan_rows": dropped_nan}


def build_rows_for_events(
    events: List[EarthquakeEvent],
    *,
    amsr2_root: Path,
    pre_days: int,
    post_days: int,
    pixels_per_event_day: int,
    control_dates_per_event: int,
    doy_window: int,
    seed: int,
    lat_limit: float,
    spec: AMSR2Spec,
    max_events: Optional[int] = None,
    all_events_for_exclusion: Optional[List[EarthquakeEvent]] = None,
):
    rng = np.random.default_rng(int(seed))
    events_use = events[: int(max_events)] if max_events else events

    all_events = all_events_for_exclusion or events
    excluded_dates = build_excluded_dates(
        [(ev.event_date, int(pre_days), int(post_days)) for ev in all_events]
    )
    years = [ev.event_date.year for ev in all_events]
    min_year, max_year = (min(years), max(years)) if years else (2013, 2023)

    # 为了减少重复IO：按日期处理
    tasks_by_date: Dict[date, List[dict]] = {}
    for ev in events_use:
        r_km = dobrovolsky_radius_km(ev.magnitude)
        # 真实事件窗（flag=1）
        for day_offset, sample_day in _iter_window_days(ev.event_date, pre_days, post_days):
            tasks_by_date.setdefault(sample_day, []).append(
                {
                    "event": ev,
                    "radius_km": r_km,
                    "day_offset": int(day_offset),
                    "sample_day": sample_day,
                    "flag": 1,
                    "window_type": "event",
                    "anchor_date": ev.event_date,
                    "control_index": None,
                }
            )

        # 对照窗（flag=0）：固定地点+季节（DOY），随机抽取不与任何真实事件窗重叠的日期
        if int(control_dates_per_event) > 0:
            controls = sample_control_dates(
                event_date=ev.event_date,
                n_controls=int(control_dates_per_event),
                rng=rng,
                min_year=int(min_year),
                max_year=int(max_year),
                doy_window=int(doy_window),
                excluded_dates=excluded_dates,
                pre_days=int(pre_days),
                post_days=int(post_days),
            )
            for c_idx, anchor in enumerate(controls):
                for day_offset, sample_day in _iter_window_days(anchor, pre_days, post_days):
                    tasks_by_date.setdefault(sample_day, []).append(
                        {
                            "event": ev,
                            "radius_km": r_km,
                            "day_offset": int(day_offset),
                            "sample_day": sample_day,
                            "flag": 0,
                            "window_type": "control",
                            "anchor_date": anchor,
                            "control_index": int(c_idx),
                        }
                    )

    all_days = sorted(tasks_by_date.keys())
    logger.info("待处理日期数: %d", len(all_days))

    for idx, day in enumerate(all_days, start=1):
        tasks = tasks_by_date[day]
        logger.info("(%d/%d) 处理日期 %s（tasks=%d）", idx, len(all_days), day.isoformat(), len(tasks))

        try:
            with AMSR2DailyGrid(amsr2_root, day, spec=spec) as grid:
                for t in tasks:
                    try:
                        ev: EarthquakeEvent = t["event"]
                        radius_km = float(t["radius_km"])
                        day_offset = int(t["day_offset"])
                        flag = int(t.get("flag", 1))
                        window_type = str(t.get("window_type", "event"))
                        anchor_date = t.get("anchor_date", ev.event_date)
                        control_index = t.get("control_index", None)

                        # 采样像元（按任务分别采样/读取，保持空间局部性）
                        pts = sample_points_within_radius(
                            event_lat=ev.latitude,
                            event_lon=ev.longitude,
                            radius_km=radius_km,
                            n_points=int(pixels_per_event_day),
                            rng=rng,
                            lat_limit=float(lat_limit),
                        )

                        feats = grid.read_features_at(pts.grid_i.astype(np.int64), pts.grid_j.astype(np.int64))

                        # 组装行（逐点 yield）
                        zone_type = int(ev.zone_type) if ev.zone_type is not None else -1
                        label = zone_type + 5 * flag if zone_type >= 0 else -1

                        n = int(pts.grid_i.size)
                        for k in range(n):
                            row = {
                                "event_id": ev.event_id,
                                "event_date": ev.event_date.isoformat(),
                                "event_lat": float(ev.latitude),
                                "event_lon": float(ev.longitude),
                                "event_mag": float(ev.magnitude),
                                "event_depth_km": float(ev.depth_km),
                                "zone_type": zone_type,
                                "flag": int(flag),
                                "label": int(label),
                                "window_type": window_type,
                                "anchor_date": anchor_date.isoformat()
                                if hasattr(anchor_date, "isoformat")
                                else str(anchor_date),
                                "control_index": (-1 if control_index is None else int(control_index)),
                                "sample_date": day.isoformat(),
                                "day_offset": int(day_offset),
                                "grid_i": int(pts.grid_i[k]),
                                "grid_j": int(pts.grid_j[k]),
                                "pixel_lat": float(pts.pixel_lat[k]),
                                "pixel_lon": float(pts.pixel_lon[k]),
                            }
                            for col in FEATURE_COLUMNS:
                                row[col] = float(feats[col][k]) if feats[col].size > 0 else float("nan")
                            yield row
                    except Exception:
                        # 单任务失败不应导致整体数据集构建中断（保持可复现/可审计：日志中保留堆栈）
	                        logger.exception(
	                            "任务失败，跳过：day=%s event_id=%s window_type=%s flag=%s",
	                            day.isoformat(),
	                            getattr(t.get("event", None), "event_id", ""),
	                            str(t.get("window_type", "")),
	                            str(t.get("flag", "")),
	                        )
	                        continue
        except FileNotFoundError as e:
            logger.warning("缺少AMSR2文件，跳过该日: %s", e)
            continue
        except Exception:
            # 例如：文件损坏、HDF5 结构异常等
            logger.exception("处理日期失败，跳过该日: %s", day.isoformat())
            continue


def parse_args():
    repo = get_repo_paths()
    default_catalog = repo.weftt_root / "data" / "raw" / "earthquake_catalog.csv"
    default_out = repo.eval_root / "datasets" / "mbt_eval_samples_v1.parquet"

    p = argparse.ArgumentParser(description="Build evaluation samples from AMSR2 HDF5 grids")
    p.add_argument("--catalog_csv", type=str, default=str(default_catalog), help="earthquake_catalog.csv 路径")
    p.add_argument("--amsr2_root", type=str, required=True, help="AMSR2 原始数据根目录（含 06-25/10-25/...）")
    p.add_argument("--out_path", type=str, default=str(default_out), help="输出 Parquet 路径")
    p.add_argument("--min_mag", type=float, default=7.0)
    p.add_argument(
        "--event_ids",
        type=str,
        default=None,
        help="可选：仅使用指定 event_id（逗号分隔，或传入一个包含event_id的文本文件路径）",
    )
    p.add_argument("--pre_days", type=int, default=20)
    p.add_argument("--post_days", type=int, default=10)
    p.add_argument("--pixels_per_event_day", type=int, default=2000)
    p.add_argument("--control_dates_per_event", type=int, default=0, help="每个事件采样多少个对照窗（flag=0）")
    p.add_argument("--doy_window", type=int, default=15, help="对照日期抽样时，控制同季节（±doy_window 天）")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lat_limit", type=float, default=70.0, help="仅在|lat|<=lat_limit范围内采样")
    p.add_argument("--max_events", type=int, default=None, help="仅取前N个事件（用于PoC）")
    p.add_argument("--log_file", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging(log_file=args.log_file)

    catalog_csv = resolve_path(args.catalog_csv)
    amsr2_root = Path(args.amsr2_root)
    out_path = resolve_path(args.out_path)

    events = load_events(catalog_csv, min_mag=args.min_mag, require_zone=True)
    logger.info("筛选事件数（M>=%.1f）：%d", args.min_mag, len(events))

    # 允许按 event_id 过滤（用于可控评估子集）
    if args.event_ids:
        raw = str(args.event_ids).strip()
        ids: List[str]
        p = Path(raw)
        if p.exists():
            ids = [x.strip() for x in p.read_text(encoding="utf-8", errors="replace").splitlines() if x.strip()]
        else:
            ids = [x.strip() for x in raw.split(",") if x.strip()]
        ids_set = set(ids)
        before = len(events)
        events = [e for e in events if e.event_id in ids_set]
        logger.info("按 event_ids 过滤: %d -> %d", before, len(events))

    spec = AMSR2Spec(orbit_tag="EQMD", mean_tag="01D")

    rows_iter = build_rows_for_events(
        events,
        amsr2_root=amsr2_root,
        pre_days=args.pre_days,
        post_days=args.post_days,
        pixels_per_event_day=args.pixels_per_event_day,
        control_dates_per_event=args.control_dates_per_event,
        doy_window=args.doy_window,
        seed=args.seed,
        lat_limit=args.lat_limit,
        spec=spec,
        max_events=args.max_events,
        all_events_for_exclusion=events,
    )

    stats = _write_parquet_stream(out_path, rows_iter)
    logger.info("写入完成: %s", out_path)
    logger.info("统计: %s", stats)

    # 记录一次构建参数，便于审计
    meta_path = out_path.with_suffix(".meta.json")
    write_json(
        meta_path,
        {
            "generated_at_utc": datetime.utcnow().isoformat() + "Z",
            "catalog_csv": str(catalog_csv),
            "amsr2_root": str(amsr2_root),
            "out_path": str(out_path),
            "min_mag": float(args.min_mag),
            "event_ids": (str(args.event_ids) if args.event_ids else None),
            "pre_days": int(args.pre_days),
            "post_days": int(args.post_days),
            "pixels_per_event_day": int(args.pixels_per_event_day),
            "control_dates_per_event": int(args.control_dates_per_event),
            "doy_window": int(args.doy_window),
            "seed": int(args.seed),
            "lat_limit": float(args.lat_limit),
            "max_events": (int(args.max_events) if args.max_events else None),
            "amsr2_spec": asdict(spec),
            "write_stats": stats,
        },
    )
    logger.info("元信息: %s", meta_path)


if __name__ == "__main__":
    main()
