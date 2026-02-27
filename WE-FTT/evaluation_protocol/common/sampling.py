from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from .geo import (
    BBox,
    GRID_N_LAT,
    GRID_N_LON,
    bbox_for_radius,
    grid_lat,
    grid_lon,
    haversine_km,
    index_ranges_for_bbox,
)


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SampledPoints:
    grid_i: np.ndarray  # int32
    grid_j: np.ndarray  # int32
    pixel_lat: np.ndarray  # float32
    pixel_lon: np.ndarray  # float32


def sample_points_within_radius(
    *,
    event_lat: float,
    event_lon: float,
    radius_km: float,
    n_points: int,
    rng: np.random.Generator,
    lat_limit: float = 70.0,
    max_attempts_factor: int = 200,
) -> SampledPoints:
    """
    在以 event 为圆心、radius_km 为半径的区域内，基于 0.25°网格进行拒绝采样。
    """
    if n_points <= 0:
        return SampledPoints(
            grid_i=np.array([], dtype=np.int32),
            grid_j=np.array([], dtype=np.int32),
            pixel_lat=np.array([], dtype=np.float32),
            pixel_lon=np.array([], dtype=np.float32),
        )

    bbox = bbox_for_radius(event_lat, event_lon, radius_km)
    # 限制在有效纬度范围（与主文一致：-70~70）
    bbox = BBox(
        lat_min=max(bbox.lat_min, -float(lat_limit)),
        lat_max=min(bbox.lat_max, float(lat_limit)),
        lon_min=bbox.lon_min,
        lon_max=bbox.lon_max,
    )

    (i_min, i_max), j_ranges = index_ranges_for_bbox(bbox)
    if i_min == i_max and i_min in (0, GRID_N_LAT - 1):
        logger.warning("bbox 可能落在边界区域，采样有效率可能很低")

    # 经度范围长度用于分段按比例采样
    seg_lens = np.array([j1 - j0 + 1 for (j0, j1) in j_ranges], dtype=np.int64)
    seg_probs = seg_lens / seg_lens.sum()

    max_attempts = max(n_points * max_attempts_factor, 10_000)
    acc_i: List[int] = []
    acc_j: List[int] = []

    attempts = 0
    while len(acc_i) < n_points and attempts < max_attempts:
        attempts += 1
        i = int(rng.integers(i_min, i_max + 1))
        seg = int(rng.choice(len(j_ranges), p=seg_probs))
        j0, j1 = j_ranges[seg]
        j = int(rng.integers(j0, j1 + 1))

        plat = grid_lat(i)
        plon = grid_lon(j)
        if abs(plat) > lat_limit:
            continue
        d = haversine_km(event_lat, event_lon, plat, plon)
        if d <= radius_km:
            acc_i.append(i)
            acc_j.append(j)

    if len(acc_i) < n_points:
        raise RuntimeError(f"采样失败：仅获得 {len(acc_i)}/{n_points} 点（attempts={attempts}）")

    i_arr = np.asarray(acc_i, dtype=np.int32)
    j_arr = np.asarray(acc_j, dtype=np.int32)
    lat_arr = np.asarray([grid_lat(int(x)) for x in i_arr], dtype=np.float32)
    lon_arr = np.asarray([grid_lon(int(x)) for x in j_arr], dtype=np.float32)
    return SampledPoints(grid_i=i_arr, grid_j=j_arr, pixel_lat=lat_arr, pixel_lon=lon_arr)


def date_window(center: date, pre_days: int, post_days: int) -> List[date]:
    return [center + timedelta(days=d) for d in range(-int(pre_days), int(post_days) + 1)]


def build_excluded_dates(events: Iterable[Tuple[date, int, int]]) -> Set[date]:
    """
    输入 (event_date, pre_days, post_days) 序列，构建所有窗口覆盖日期集合。
    """
    excluded: Set[date] = set()
    for d, pre, post in events:
        for x in date_window(d, pre, post):
            excluded.add(x)
    return excluded


def sample_control_dates(
    *,
    event_date: date,
    n_controls: int,
    rng: np.random.Generator,
    min_year: int,
    max_year: int,
    doy_window: int,
    excluded_dates: Set[date],
    pre_days: int,
    post_days: int,
    max_tries: int = 10_000,
) -> List[date]:
    """
    采样对照“伪事件日期”：
    - 控制季节性：同月同日 ± doy_window 天，并允许跨年随机
    - 避免与任何真实事件窗口重叠（以 excluded_dates 为准）
    """
    controls: List[date] = []
    tries = 0

    def safe_date(y: int, m: int, d: int) -> date:
        # 处理 2/29 等非法日期：回退到 2/28
        try:
            return date(y, m, d)
        except Exception:
            if m == 2 and d == 29:
                return date(y, 2, 28)
            # 最保守回退
            return date(y, m, min(d, 28))

    while len(controls) < n_controls and tries < max_tries:
        tries += 1
        y = int(rng.integers(min_year, max_year + 1))
        base = safe_date(y, event_date.month, event_date.day)
        jitter = int(rng.integers(-int(doy_window), int(doy_window) + 1))
        cand = base + timedelta(days=jitter)

        # 约束：cand 的整个窗口不与 excluded_dates 相交
        ok = True
        for x in date_window(cand, pre_days, post_days):
            if x in excluded_dates:
                ok = False
                break
        if not ok:
            continue
        controls.append(cand)

    if len(controls) < n_controls:
        raise RuntimeError(f"对照日期采样失败：{len(controls)}/{n_controls}（tries={tries}）")
    return controls

