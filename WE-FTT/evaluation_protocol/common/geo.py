from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple


# AMSR2-L3 EQR 0.25° 网格（720×1440）
GRID_STEP_DEG = 0.25
GRID_N_LAT = 720
GRID_N_LON = 1440
GRID_LAT0 = 89.875   # i=0
GRID_LON0 = -179.875 # j=0


def normalize_lon(lon: float) -> float:
    """归一化到 [-180, 180)"""
    x = (lon + 180.0) % 360.0 - 180.0
    # 保证 180 映射到 -180
    return -180.0 if x == 180.0 else x


def grid_lat(i: int) -> float:
    return GRID_LAT0 - i * GRID_STEP_DEG


def grid_lon(j: int) -> float:
    return GRID_LON0 + j * GRID_STEP_DEG


def lat_to_i(lat: float) -> int:
    """将纬度映射到最近网格索引 i（0..719）。"""
    i = int(round((GRID_LAT0 - lat) / GRID_STEP_DEG))
    return max(0, min(GRID_N_LAT - 1, i))


def lon_to_j(lon: float) -> int:
    """将经度映射到最近网格索引 j（0..1439）。"""
    lon_n = normalize_lon(lon)
    j = int(round((lon_n - GRID_LON0) / GRID_STEP_DEG))
    j = j % GRID_N_LON
    return j


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """球面距离（km）。"""
    r = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(normalize_lon(lon2 - lon1))
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def dobrovolsky_radius_km(magnitude: float) -> float:
    """Dobrovolsky 半径：R = 10^(0.43M) km"""
    return 10 ** (0.43 * float(magnitude))


@dataclass(frozen=True)
class BBox:
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float


def bbox_for_radius(lat: float, lon: float, radius_km: float) -> BBox:
    """
    以近似方式将 km 半径转为经纬度 bbox（用于快速候选采样）。
    """
    dlat = radius_km / 111.32
    # 避免 cos(lat)=0
    dlon = radius_km / (111.32 * max(1e-6, math.cos(math.radians(lat))))
    lat_min = max(-90.0, lat - dlat)
    lat_max = min(90.0, lat + dlat)
    lon_min = normalize_lon(lon - dlon)
    lon_max = normalize_lon(lon + dlon)
    return BBox(lat_min=lat_min, lat_max=lat_max, lon_min=lon_min, lon_max=lon_max)


def index_ranges_for_bbox(b: BBox) -> Tuple[Tuple[int, int], List[Tuple[int, int]]]:
    """
    将 bbox 映射到网格索引范围：
    - 纬度返回 (i_min, i_max)（包含端点）
    - 经度返回若干段 (j_min, j_max)（包含端点，处理跨日界线）
    """
    # 纬度：注意网格是从北到南
    i_min = int(math.floor((GRID_LAT0 - b.lat_max) / GRID_STEP_DEG))
    i_max = int(math.ceil((GRID_LAT0 - b.lat_min) / GRID_STEP_DEG))
    i_min = max(0, min(GRID_N_LAT - 1, i_min))
    i_max = max(0, min(GRID_N_LAT - 1, i_max))
    if i_min > i_max:
        i_min, i_max = i_max, i_min

    # 经度：网格从 -179.875 到 179.875
    def lon_to_j_floor(x: float) -> int:
        return int(math.floor((normalize_lon(x) - GRID_LON0) / GRID_STEP_DEG)) % GRID_N_LON

    def lon_to_j_ceil(x: float) -> int:
        return int(math.ceil((normalize_lon(x) - GRID_LON0) / GRID_STEP_DEG)) % GRID_N_LON

    j0 = lon_to_j_floor(b.lon_min)
    j1 = lon_to_j_ceil(b.lon_max)

    if b.lon_min <= b.lon_max:
        # 不跨日界线
        return (i_min, i_max), [(j0, j1)]

    # 跨日界线：分两段 [lon_min,180) U [-180, lon_max]
    return (i_min, i_max), [(j0, GRID_N_LON - 1), (0, j1)]

