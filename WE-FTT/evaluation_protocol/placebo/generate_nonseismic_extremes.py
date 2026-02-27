#!/usr/bin/env python3
"""
生成 nonseismic_extreme placebo 计划（jsonl，v1）。

定义（ERA5 代理）：
- 在 source_event 固定位置上，选择满足 ERA5 极端阈值的“非震日期”作为 anchor_date：
  - wind_speed >= wind_thresh_mps 或 precip_mm_day >= precip_thresh_mm_day
- 非震约束：anchor_date 不得落入任意真实地震窗口扩展区间
  [event_date-(pre_days+min_gap), event_date+(post_days+min_gap)]

输出行字段与 run_placebo.py 兼容：
- placebo_type 固定为 nonseismic_extreme
- anchor_lat/anchor_lon 采用 source_event 的经纬度（隔离“日期-天气”影响）
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from dataclasses import asdict
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# 将 WE-FTT 根目录加入路径（保证可直接以脚本方式运行）
weftt_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if weftt_root not in sys.path:
    sys.path.insert(0, weftt_root)

from evaluation_protocol.common.catalog import EarthquakeEvent, load_events  # noqa: E402
from evaluation_protocol.common.geo import lat_to_i, lon_to_j  # noqa: E402
from evaluation_protocol.common.jsonl import write_json, write_jsonl  # noqa: E402
from evaluation_protocol.common.logging_utils import setup_logging  # noqa: E402
from evaluation_protocol.common.paths import get_repo_paths, resolve_path  # noqa: E402
from evaluation_protocol.common.sampling import build_excluded_dates  # noqa: E402
from evaluation_protocol.common.splits import load_event_splits  # noqa: E402


logger = logging.getLogger(__name__)


def _list_cache_days(cache_dir: Path) -> List[date]:
    pat = re.compile(r"^(wind_speed|precip_mm)_(\d{8})\.npy$")
    wind_days: set[str] = set()
    precip_days: set[str] = set()
    for fp in cache_dir.glob("*.npy"):
        m = pat.match(fp.name)
        if not m:
            continue
        prefix, ymd = m.group(1), m.group(2)
        if prefix == "wind_speed":
            wind_days.add(ymd)
        else:
            precip_days.add(ymd)
    both = sorted(wind_days & precip_days)
    out: List[date] = []
    for ymd in both:
        out.append(date(int(ymd[0:4]), int(ymd[4:6]), int(ymd[6:8])))
    return out


def _load_day(cache_dir: Path, day: date) -> Tuple[np.ndarray, np.ndarray]:
    ymd = day.strftime("%Y%m%d")
    ws = np.load(cache_dir / f"wind_speed_{ymd}.npy")
    pr = np.load(cache_dir / f"precip_mm_{ymd}.npy")
    return ws, pr


def _events_by_id(events: Sequence[EarthquakeEvent]) -> Dict[str, EarthquakeEvent]:
    return {e.event_id: e for e in events}


def _select_split_ids(*, split_name: str, splits_path: Path) -> List[str]:
    splits = load_event_splits(splits_path)
    if split_name == "train":
        return sorted(splits.train_event_ids)
    if split_name == "val":
        return sorted(splits.val_event_ids)
    return sorted(splits.test_event_ids)


def _filter_zone(events: Sequence[EarthquakeEvent], zone_filter: str) -> List[EarthquakeEvent]:
    zf = zone_filter.strip().lower()
    if zf == "all":
        return list(events)
    if zf == "land":
        return [e for e in events if e.zone_type is not None and int(e.zone_type) in (1, 2, 3, 4)]
    if zf == "ocean":
        return [e for e in events if e.zone_type is not None and int(e.zone_type) == 0]
    raise ValueError(f"未知 zone_filter: {zone_filter}")


def _build_candidates(
    *,
    selected_events: Sequence[EarthquakeEvent],
    all_days: Sequence[date],
    cache_dir: Path,
    excluded_dates: set[date],
    wind_thresh_mps: float,
    precip_thresh_mm_day: float,
) -> Dict[str, List[Tuple[date, float, float]]]:
    candidates: Dict[str, List[Tuple[date, float, float]]] = {e.event_id: [] for e in selected_events}
    if not selected_events:
        return candidates

    ij = {e.event_id: (lat_to_i(float(e.latitude)), lon_to_j(float(e.longitude))) for e in selected_events}

    for day in all_days:
        if day in excluded_dates:
            continue
        try:
            ws, pr = _load_day(cache_dir, day)
        except FileNotFoundError:
            continue

        for e in selected_events:
            ii, jj = ij[e.event_id]
            wind = float(ws[ii, jj])
            precip = float(pr[ii, jj])
            if np.isnan(wind) and np.isnan(precip):
                continue
            if (wind >= float(wind_thresh_mps)) or (precip >= float(precip_thresh_mm_day)):
                candidates[e.event_id].append((day, wind, precip))

    return candidates


def _generate_rows(
    *,
    selected_events: Sequence[EarthquakeEvent],
    candidates: Dict[str, List[Tuple[date, float, float]]],
    n_repeats: int,
    seed: int,
    wind_thresh_mps: float,
    precip_thresh_mm_day: float,
) -> List[dict]:
    rng = np.random.default_rng(int(seed))
    rows: List[dict] = []
    for rep in range(int(n_repeats)):
        for e in selected_events:
            cand = candidates[e.event_id]
            if not cand:
                continue
            idx = int(rng.integers(0, len(cand)))
            anchor_day, wind, precip = cand[idx]
            rows.append(
                {
                    "version": "placebo_plan_v1",
                    "replicate_id": int(rep),
                    "placebo_type": "nonseismic_extreme",
                    "source_event_id": e.event_id,
                    "source_event_date": e.event_date.isoformat(),
                    "source_lat": float(e.latitude),
                    "source_lon": float(e.longitude),
                    "source_mag": float(e.magnitude),
                    "source_depth_km": float(e.depth_km),
                    "zone_type": int(e.zone_type) if e.zone_type is not None else None,
                    "anchor_date": anchor_day.isoformat(),
                    "anchor_lat": float(e.latitude),
                    "anchor_lon": float(e.longitude),
                    "extreme_proxy": "era5",
                    "extreme_wind_mps": float(wind),
                    "extreme_precip_mm_day": float(precip),
                    "extreme_wind_thresh_mps": float(wind_thresh_mps),
                    "extreme_precip_thresh_mm_day": float(precip_thresh_mm_day),
                    "candidate_days_for_event": int(len(cand)),
                }
            )
    return rows


def parse_args():
    repo = get_repo_paths()
    default_catalog = repo.weftt_root / "data" / "raw" / "earthquake_catalog.csv"
    default_splits = repo.eval_root / "data_splits" / "event_grouped_splits_v1.json"
    default_cache = repo.eval_root / "era5" / "cache" / "daily"
    default_out = repo.eval_root / "placebo" / "plans" / "nonseismic_extreme_plan_v1.jsonl"

    p = argparse.ArgumentParser(description="Generate nonseismic_extreme placebo plan (ERA5 proxy)")
    p.add_argument("--catalog_csv", type=str, default=str(default_catalog))
    p.add_argument("--splits_json", type=str, default=str(default_splits))
    p.add_argument("--era5_daily_cache_dir", type=str, default=str(default_cache))
    p.add_argument("--use_split", type=str, default="test", choices=["train", "val", "test"])
    p.add_argument("--zone_filter", type=str, default="land", choices=["all", "land", "ocean"])

    p.add_argument("--n_repeats", type=int, default=400)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pre_days", type=int, default=20)
    p.add_argument("--post_days", type=int, default=10)
    p.add_argument("--min_days_from_eq_window", type=int, default=31)
    p.add_argument("--wind_thresh_mps", type=float, default=10.0)
    p.add_argument("--precip_thresh_mm_day", type=float, default=5.0)
    p.add_argument("--min_candidates_per_event", type=int, default=1)
    p.add_argument(
        "--strict_event_coverage",
        action="store_true",
        help="若任一 source_event 可选极端日不足 min_candidates_per_event，则报错退出。",
    )

    p.add_argument("--out_jsonl", type=str, default=str(default_out))
    p.add_argument("--out_meta_json", type=str, default=None)
    p.add_argument("--log_file", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging(log_file=args.log_file)

    catalog_csv = resolve_path(args.catalog_csv)
    splits_json = resolve_path(args.splits_json)
    cache_dir = resolve_path(args.era5_daily_cache_dir)
    out_jsonl = resolve_path(args.out_jsonl)
    out_meta = (
        resolve_path(args.out_meta_json)
        if args.out_meta_json
        else out_jsonl.with_suffix(".meta.json")
    )

    if not cache_dir.exists():
        raise FileNotFoundError(f"ERA5 daily cache 不存在: {cache_dir}")

    events = load_events(catalog_csv, min_mag=7.0, require_zone=True)
    by_id = _events_by_id(events)
    selected_ids = _select_split_ids(split_name=str(args.use_split), splits_path=splits_json)
    selected = [by_id[eid] for eid in selected_ids if eid in by_id]
    selected = _filter_zone(selected, str(args.zone_filter))
    if not selected:
        raise ValueError("按 split + zone_filter 过滤后，source events 为空。")

    excluded_dates = build_excluded_dates(
        [
            (
                e.event_date,
                int(args.pre_days) + int(args.min_days_from_eq_window),
                int(args.post_days) + int(args.min_days_from_eq_window),
            )
            for e in events
        ]
    )
    all_days = _list_cache_days(cache_dir)
    if not all_days:
        raise ValueError(f"未找到可用 ERA5 daily 缓存（wind_speed/precip_mm 成对文件）：{cache_dir}")

    candidates = _build_candidates(
        selected_events=selected,
        all_days=all_days,
        cache_dir=cache_dir,
        excluded_dates=excluded_dates,
        wind_thresh_mps=float(args.wind_thresh_mps),
        precip_thresh_mm_day=float(args.precip_thresh_mm_day),
    )

    kept: List[EarthquakeEvent] = []
    dropped: List[dict] = []
    min_need = int(max(1, args.min_candidates_per_event))
    for e in selected:
        n_cand = int(len(candidates.get(e.event_id, [])))
        if n_cand >= min_need:
            kept.append(e)
        else:
            dropped.append({"event_id": e.event_id, "n_candidates": n_cand, "zone_type": int(e.zone_type)})

    if dropped and bool(args.strict_event_coverage):
        raise ValueError(f"有 {len(dropped)} 个事件候选极端日不足 min_candidates_per_event={min_need}，严格模式退出。")
    if not kept:
        raise ValueError("所有事件均无可用非震极端候选日，无法生成 plan。")

    rows = _generate_rows(
        selected_events=kept,
        candidates=candidates,
        n_repeats=int(args.n_repeats),
        seed=int(args.seed),
        wind_thresh_mps=float(args.wind_thresh_mps),
        precip_thresh_mm_day=float(args.precip_thresh_mm_day),
    )
    if not rows:
        raise ValueError("生成结果为空，请检查阈值或缓存覆盖日期。")

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_jsonl, rows)

    meta = {
        "version": "nonseismic_extreme_plan_v1",
        "generated_at_utc": datetime.utcnow().isoformat() + "Z",
        "inputs": {
            "catalog_csv": str(catalog_csv),
            "splits_json": str(splits_json),
            "era5_daily_cache_dir": str(cache_dir),
            "use_split": str(args.use_split),
            "zone_filter": str(args.zone_filter),
        },
        "params": {
            "n_repeats": int(args.n_repeats),
            "seed": int(args.seed),
            "pre_days": int(args.pre_days),
            "post_days": int(args.post_days),
            "min_days_from_eq_window": int(args.min_days_from_eq_window),
            "wind_thresh_mps": float(args.wind_thresh_mps),
            "precip_thresh_mm_day": float(args.precip_thresh_mm_day),
            "min_candidates_per_event": int(min_need),
            "strict_event_coverage": bool(args.strict_event_coverage),
        },
        "stats": {
            "cache_days_total": int(len(all_days)),
            "source_events_selected": int(len(selected)),
            "source_events_kept": int(len(kept)),
            "source_events_dropped": dropped,
            "rows_written": int(len(rows)),
        },
        "artifacts": {"plan_jsonl": str(out_jsonl)},
    }
    write_json(out_meta, meta)
    logger.info(
        "写入 nonseismic_extreme 计划：rows=%d kept_events=%d dropped=%d -> %s",
        int(len(rows)),
        int(len(kept)),
        int(len(dropped)),
        out_jsonl,
    )


if __name__ == "__main__":
    main()

