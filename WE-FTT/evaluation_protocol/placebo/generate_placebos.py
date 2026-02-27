#!/usr/bin/env python3
"""
生成 placebo “计划文件”（jsonl），用于后续可复现地构造 placebo 测试集（v1）。

说明：
- 本脚本只生成“锚点”（日期/位置）与必要元信息，不读取 AMSR2。
- 具体样本生成与模型评估在 `run_placebo.py` 中完成。
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

# 将 WE-FTT 根目录加入路径（保证可直接以脚本方式运行）
weftt_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if weftt_root not in sys.path:
    sys.path.insert(0, weftt_root)

from evaluation_protocol.common.catalog import EarthquakeEvent, load_events  # noqa: E402
from evaluation_protocol.common.geo import haversine_km  # noqa: E402
from evaluation_protocol.common.jsonl import write_jsonl  # noqa: E402
from evaluation_protocol.common.logging_utils import setup_logging  # noqa: E402
from evaluation_protocol.common.paths import get_repo_paths, resolve_path  # noqa: E402
from evaluation_protocol.common.sampling import build_excluded_dates, sample_control_dates  # noqa: E402
from evaluation_protocol.common.splits import load_event_splits  # noqa: E402


def _events_by_id(events: Sequence[EarthquakeEvent]) -> Dict[str, EarthquakeEvent]:
    return {e.event_id: e for e in events}


def _sample_random_location_donors(
    source: EarthquakeEvent,
    candidates: Sequence[EarthquakeEvent],
    *,
    rng: np.random.Generator,
    min_distance_km: float,
    donor_same_hemisphere: bool,
    donor_max_lat_diff_deg: Optional[float],
) -> tuple[EarthquakeEvent, dict]:
    """
    在同 zone_type 的事件中，采样一个“足够远”的 donor 作为随机位置 placebo。
    """
    if not candidates:
        raise RuntimeError("random_location：候选 donor 为空。")

    def same_hemisphere(a_lat: float, b_lat: float) -> bool:
        # 允许赤道（lat=0）落到任一侧，避免边界问题
        if a_lat == 0.0 or b_lat == 0.0:
            return True
        return (a_lat > 0.0) == (b_lat > 0.0)

    def eligible(
        *,
        require_same_hemi: bool,
        max_lat_diff: Optional[float],
    ) -> list[EarthquakeEvent]:
        out: list[EarthquakeEvent] = []
        for donor in candidates:
            if donor.event_id == source.event_id:
                continue
            if require_same_hemi and not same_hemisphere(float(source.latitude), float(donor.latitude)):
                continue
            if max_lat_diff is not None and abs(float(source.latitude) - float(donor.latitude)) > float(max_lat_diff):
                continue
            d_km = haversine_km(source.latitude, source.longitude, donor.latitude, donor.longitude)
            if d_km < float(min_distance_km):
                continue
            out.append(donor)
        return out

    # 先按“同半球 + 近纬度带”筛选；若筛选后为空，则逐级放宽（并记录到 plan）
    relaxed_lat = False
    relaxed_hemisphere = False

    elig = eligible(require_same_hemi=bool(donor_same_hemisphere), max_lat_diff=donor_max_lat_diff_deg)
    if not elig and donor_max_lat_diff_deg is not None:
        relaxed_lat = True
        elig = eligible(require_same_hemi=bool(donor_same_hemisphere), max_lat_diff=None)
    if not elig and bool(donor_same_hemisphere):
        relaxed_hemisphere = True
        elig = eligible(require_same_hemi=False, max_lat_diff=None)
    if not elig:
        raise RuntimeError("random_location：未找到满足距离/环境约束的 donor。")

    donor = elig[int(rng.integers(0, len(elig)))]
    d_km = haversine_km(source.latitude, source.longitude, donor.latitude, donor.longitude)
    lat_diff = abs(float(source.latitude) - float(donor.latitude))
    cross = not same_hemisphere(float(source.latitude), float(donor.latitude))
    meta = {
        "donor_distance_km": float(d_km),
        "donor_lat_diff_deg": float(lat_diff),
        "donor_cross_hemisphere": bool(cross),
        "donor_same_hemisphere_requested": bool(donor_same_hemisphere),
        "donor_max_lat_diff_deg_requested": None if donor_max_lat_diff_deg is None else float(donor_max_lat_diff_deg),
        "donor_relaxed_lat_constraint": bool(relaxed_lat),
        "donor_relaxed_hemisphere_constraint": bool(relaxed_hemisphere),
        "donor_eligible_count": int(len(elig)),
    }
    return donor, meta


def generate_plan_rows(
    events: Sequence[EarthquakeEvent],
    *,
    event_ids: Sequence[str],
    placebo_type: str,
    n_repeats: int,
    seed: int,
    pre_days: int,
    post_days: int,
    doy_window: int,
    time_shifts: Sequence[int],
    min_distance_km: float,
    donor_same_hemisphere: bool,
    donor_max_lat_diff_deg: Optional[float],
) -> List[dict]:
    rng = np.random.default_rng(int(seed))
    by_id = _events_by_id(events)

    selected: List[EarthquakeEvent] = []
    for eid in event_ids:
        if eid in by_id:
            selected.append(by_id[eid])

    excluded_dates = build_excluded_dates([(e.event_date, int(pre_days), int(post_days)) for e in events])
    years = [e.event_date.year for e in events]
    min_year, max_year = (min(years), max(years)) if years else (2013, 2023)

    rows: List[dict] = []
    placebo_type = str(placebo_type).strip().lower()
    if placebo_type not in {"random_date", "time_shift", "random_location"}:
        raise ValueError(f"不支持的 placebo_type: {placebo_type}")

    # random_location: 仅在同 zone_type 中采样 donor
    donors_by_zone: Dict[int, List[EarthquakeEvent]] = {}
    if placebo_type == "random_location":
        for e in events:
            if e.zone_type is None:
                continue
            donors_by_zone.setdefault(int(e.zone_type), []).append(e)

    for rep in range(int(n_repeats)):
        for e in selected:
            if e.zone_type is None:
                continue
            z = int(e.zone_type)
            anchor_date = e.event_date
            anchor_lat = float(e.latitude)
            anchor_lon = float(e.longitude)
            donor_id: Optional[str] = None

            if placebo_type == "random_date":
                # 同地点，同季节随机日期（避开所有真实事件窗）
                [anchor_date] = sample_control_dates(
                    event_date=e.event_date,
                    n_controls=1,
                    rng=rng,
                    min_year=int(min_year),
                    max_year=int(max_year),
                    doy_window=int(doy_window),
                    excluded_dates=excluded_dates,
                    pre_days=int(pre_days),
                    post_days=int(post_days),
                )
            elif placebo_type == "time_shift":
                # 同地点，时间平移（若落入 excluded_dates，则在 shift 周围做小幅抖动重试）
                shift = int(time_shifts[int(rng.integers(0, len(time_shifts)))])
                base = e.event_date + timedelta(days=shift)
                # 抖动：优先尝试 0, ±1, ±2, ... ±doy_window
                ok = False
                for j in [0, *sum(([k, -k] for k in range(1, int(doy_window) + 1)), [])]:
                    cand = base + timedelta(days=int(j))
                    # 约束：cand 的整个窗口不与 excluded_dates 相交
                    bad = False
                    for d in range(-int(pre_days), int(post_days) + 1):
                        if (cand + timedelta(days=d)) in excluded_dates:
                            bad = True
                            break
                    if not bad:
                        anchor_date = cand
                        ok = True
                        break
                if not ok:
                    # 退化：仍使用 base（可审计）
                    anchor_date = base
            else:  # random_location
                donors = donors_by_zone.get(z, [])
                donor, donor_meta = _sample_random_location_donors(
                    e,
                    donors,
                    rng=rng,
                    min_distance_km=float(min_distance_km),
                    donor_same_hemisphere=bool(donor_same_hemisphere),
                    donor_max_lat_diff_deg=donor_max_lat_diff_deg,
                )
                donor_id = donor.event_id
                anchor_date = e.event_date
                anchor_lat = float(donor.latitude)
                anchor_lon = float(donor.longitude)

            rows.append(
                {
                    "version": "placebo_plan_v1",
                    "replicate_id": int(rep),
                    "placebo_type": placebo_type,
                    "source_event_id": e.event_id,
                    "source_event_date": e.event_date.isoformat(),
                    "source_lat": float(e.latitude),
                    "source_lon": float(e.longitude),
                    "source_mag": float(e.magnitude),
                    "source_depth_km": float(e.depth_km),
                    "zone_type": int(e.zone_type),
                    "donor_event_id": donor_id,
                    **(donor_meta if placebo_type == "random_location" else {}),
                    "anchor_date": anchor_date.isoformat(),
                    "anchor_lat": float(anchor_lat),
                    "anchor_lon": float(anchor_lon),
                }
            )

    return rows


def parse_args():
    repo = get_repo_paths()
    default_catalog = repo.weftt_root / "data" / "raw" / "earthquake_catalog.csv"
    default_splits = repo.eval_root / "data_splits" / "event_grouped_splits_v1.json"
    default_out = repo.eval_root / "placebo" / "plans" / "placebo_plan_v1.jsonl"

    p = argparse.ArgumentParser(description="Generate placebo plans (jsonl)")
    p.add_argument("--catalog_csv", type=str, default=str(default_catalog))
    p.add_argument("--splits_json", type=str, default=str(default_splits))
    p.add_argument("--use_split", type=str, default="test", choices=["train", "val", "test"])
    p.add_argument("--placebo_type", type=str, required=True, choices=["random_date", "time_shift", "random_location"])
    p.add_argument("--n_repeats", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pre_days", type=int, default=20)
    p.add_argument("--post_days", type=int, default=10)
    p.add_argument("--doy_window", type=int, default=15)
    p.add_argument("--time_shifts", type=str, default="90,180,365", help="time_shift 的候选平移天数列表")
    p.add_argument("--min_distance_km", type=float, default=500.0)
    p.add_argument(
        "--donor_same_hemisphere",
        action="store_true",
        help="random_location: donor 与 source 同半球（默认不启用；用于避免季节翻转带来的伪信号）",
    )
    p.add_argument(
        "--donor_max_lat_diff_deg",
        type=float,
        default=None,
        help="random_location: donor 与 source 的纬度差上限（度）。默认不限制；若筛选为空会自动放宽并在 plan 中记录。",
    )
    p.add_argument("--out_jsonl", type=str, default=str(default_out))
    p.add_argument("--log_file", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging(log_file=args.log_file)

    catalog_csv = resolve_path(args.catalog_csv)
    splits_json = resolve_path(args.splits_json)
    out_jsonl = resolve_path(args.out_jsonl)

    events = load_events(catalog_csv, min_mag=7.0, require_zone=True)
    splits = load_event_splits(splits_json)
    if args.use_split == "train":
        event_ids = sorted(splits.train_event_ids)
    elif args.use_split == "val":
        event_ids = sorted(splits.val_event_ids)
    else:
        event_ids = sorted(splits.test_event_ids)

    time_shifts = [int(x) for x in str(args.time_shifts).split(",") if str(x).strip() != ""]
    if not time_shifts:
        raise ValueError("time_shifts 为空。")

    rows = generate_plan_rows(
        events,
        event_ids=event_ids,
        placebo_type=str(args.placebo_type),
        n_repeats=int(args.n_repeats),
        seed=int(args.seed),
        pre_days=int(args.pre_days),
        post_days=int(args.post_days),
        doy_window=int(args.doy_window),
        time_shifts=time_shifts,
        min_distance_km=float(args.min_distance_km),
        donor_same_hemisphere=bool(args.donor_same_hemisphere),
        donor_max_lat_diff_deg=args.donor_max_lat_diff_deg,
    )

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_jsonl, rows)


if __name__ == "__main__":
    main()
