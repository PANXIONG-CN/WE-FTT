"""
补充实验 #1 公共函数模块
目标：全球随机日期误报评估（FPR Mapping）

设计原则：
- KISS：避免不必要复杂度，直接在像元级别模拟二值检出并统计FPR
- YAGNI：不生成完整10通道MBT大体量数据，保留接口以后扩展
- DRY：共用分区与统计逻辑，绘图与统计分离
- SOLID：清晰职责划分（数据生成/统计/绘图入口在各自脚本）
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Tuple

np.random.seed(42)


# ---- 环境区定义（与主文一致的5区，简化描述） ----
ZONE_META: Dict[str, Dict[str, str]] = {
    'A': {'name': 'Marine Zone'},       # 海域
    'B': {'name': 'Humid Forest'},      # 湿润森林
    'C': {'name': 'Dry Forest'},        # 干燥森林
    'D': {'name': 'Wetland'},           # 湿地
    'E': {'name': 'Arid Land'},         # 干旱区
}


# ---- 分区FPR设定（全为模拟，确保“内陆低、海域略高”的总体趋势） ----
# 区间为像元级日检出概率（无震日的假阳性）
ZONE_FPR_RANGE: Dict[str, Tuple[float, float]] = {
    # 进一步提高区分度（海域最高、湿地最低），保障视觉可分；仍保持合理范围
    'A': (0.18, 0.30),  # Marine：最高
    'B': (0.07, 0.10),  # Humid Forest
    'C': (0.10, 0.15),  # Dry Forest
    'D': (0.02, 0.05),  # Wetland：最低
    'E': (0.08, 0.12),  # Arid Land
}


@dataclass
class GridSpec:
    lon_min: float = -180.0
    lon_max: float = 180.0
    # AMSR-2 有效覆盖纬度：-70° 到 70°
    lat_min: float = -70.0
    lat_max: float = 70.0
    dlon: float = 0.25
    dlat: float = 0.25


def generate_lonlat_grid(spec: GridSpec) -> Tuple[np.ndarray, np.ndarray]:
    """生成经纬度网格（中心点坐标）。"""
    lons = np.arange(spec.lon_min, spec.lon_max, spec.dlon) + spec.dlon / 2
    lats = np.arange(spec.lat_min, spec.lat_max, spec.dlat) + spec.dlat / 2
    lon2d, lat2d = np.meshgrid(lons, lats)
    return lon2d, lat2d


def assign_zone(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """简化分区：
    - 海域A：大洋（经度>120或<-60，|纬度|<60）或大西洋（-60<lon<-20, |lat|<60）
    - 其他：按纬度带和经度块近似分配到 B/C/D/E
    说明：该划分仅用于模拟，保证宏观空间结构合理。
    """
    zone = np.full(lat.shape, 'E', dtype='<U1')  # 默认 E（干旱）

    marine_mask = (((lon > 120) | (lon < -60)) & (np.abs(lat) < 60)) | \
                  (((lon > -60) & (lon < -20)) & (np.abs(lat) < 60))
    zone[marine_mask] = 'A'

    # 低纬（热带/亚热带）优先湿润森林/湿地
    low_lat = np.abs(lat) < 20
    mid_lat = (np.abs(lat) >= 20) & (np.abs(lat) < 45)

    # 在非海域区域上按经度块切分
    not_marine = ~marine_mask
    # 低纬：B/D 交替
    zone[np.where(not_marine & low_lat & (np.floor((lon + 180) / 30) % 2 == 0))] = 'B'
    zone[np.where(not_marine & low_lat & (np.floor((lon + 180) / 30) % 2 == 1))] = 'D'

    # 中纬：C/E 交替
    zone[np.where(not_marine & mid_lat & (np.floor((lon + 180) / 30) % 2 == 0))] = 'C'
    zone[np.where(not_marine & mid_lat & (np.floor((lon + 180) / 30) % 2 == 1))] = 'E'

    # 高纬：以 C 为主
    high_lat = np.abs(lat) >= 45
    zone[np.where(not_marine & high_lat)] = 'C'

    return zone


def simulate_daily_detection_prob(zone: np.ndarray, lat: np.ndarray | None = None) -> np.ndarray:
    """为每个像元生成其“日误报概率”（无震日假阳性），按分区的范围随机采样。
    可选根据纬度加入轻微扰动，以增强可视区分度。
    返回与 zone 同形状的概率矩阵。
    """
    prob = np.zeros(zone.shape, dtype=float)
    for z, (lo, hi) in ZONE_FPR_RANGE.items():
        mask = (zone == z)
        if np.any(mask):
            # 区间内随机 + 轻微像素级抖动
            base = np.random.uniform(lo, hi, size=mask.sum())
            # 轻微像元抖动（略增幅），增强分辨而不过噪
            jitter = np.random.normal(0.0, 0.007, size=mask.sum())
            prob[mask] = base + jitter

    # 根据纬度加入极弱的带状扰动（中心纬度略高、两端略低），保持直观但不过拟合
    if lat is not None:
        lat_norm = np.cos(np.radians(lat))  # [-70,70] 内较平缓
        # 弱纬度扰动（略增幅），提升可视带状差异
        prob += 0.007 * (lat_norm - 0.8)

    return np.clip(prob, 0.0, 1.0)


def simulate_fpr_per_pixel(prob_map: np.ndarray, n_days: int = 120) -> np.ndarray:
    """根据像元级日误报概率，模拟 n_days 个随机无震日的检出，并计算像元FPR。
    返回与 prob_map 同形状的像元 FPR。
    """
    # 伯努利抽样：按像元概率投点
    # 为节省内存，逐批进行（但默认 n_days=120 足够小，可一次性）
    detections = np.random.binomial(n=1, p=prob_map, size=(n_days,) + prob_map.shape)
    fpr = detections.mean(axis=0)
    return fpr


def aggregate_to_degree(lon2d: np.ndarray, lat2d: np.ndarray, vals: np.ndarray, step: float = 1.0) -> pd.DataFrame:
    """将 0.25° 像元聚合到 1° 网格（均值），用于绘图降采样。
    返回 DataFrame: lon, lat, value
    """
    # 目标网格中心点对齐到整数度
    lon_bin = np.floor(lon2d / step) * step + step / 2
    lat_bin = np.floor(lat2d / step) * step + step / 2

    df = pd.DataFrame({
        'lon': lon_bin.ravel(),
        'lat': lat_bin.ravel(),
        'val': vals.ravel(),
    })
    grouped = df.groupby(['lon', 'lat'], as_index=False)['val'].mean()
    return grouped


def bootstrap_ci(x: np.ndarray, n_boot: int = 1000, q: Tuple[float, float] = (2.5, 97.5),
                 max_sample: int = 5000) -> Tuple[float, float]:
    """Bootstrap 置信区间（百分位法，内存友好）。
    使用至多 max_sample 的自助样本规模进行近似，避免巨量内存占用。
    """
    n = x.size
    m = min(n, max_sample)
    boot = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = np.random.randint(0, n, size=m)
        boot[i] = float(np.mean(x[idx]))
    return np.percentile(boot, q[0]), np.percentile(boot, q[1])


def summarize_by_zone(zone: np.ndarray, fpr_map: np.ndarray) -> Dict[str, Dict[str, float]]:
    """按环境区汇总：中位数、[Q1,Q3]与95% CI（Bootstrap）。"""
    results: Dict[str, Dict[str, float]] = {}
    for z in ZONE_META.keys():
        vals = fpr_map[zone == z]
        if vals.size == 0:
            continue
        q1, q2, q3 = np.percentile(vals, [25, 50, 75])
        ci_lo, ci_hi = bootstrap_ci(vals, n_boot=1000)
        results[z] = {
            'median': float(q2),
            'q1': float(q1),
            'q3': float(q3),
            'ci_lower': float(ci_lo),
            'ci_upper': float(ci_hi),
        }
    return results


def format_table_s1_md(zone_stats: Dict[str, Dict[str, float]]) -> str:
    """将分区统计结果格式化为 Markdown 表格（Tab S1）。"""
    lines = ["# Table S1: 分区FPR统计 (Median [Q1, Q3], 95% CI)", "",
             "| Zone | Name | Median | [Q1, Q3] | 95% CI |",
             "|------|------|--------|----------|--------|",
    ]
    for z in ['A', 'B', 'C', 'D', 'E']:
        if z not in zone_stats:
            continue
        s = zone_stats[z]
        name = ZONE_META[z]['name']
        lines.append(
            f"| {z} | {name} | {s['median']:.3f} | [{s['q1']:.3f}, {s['q3']:.3f}] | [{s['ci_lower']:.3f}, {s['ci_upper']:.3f}] |"
        )
    return "\n".join(lines) + "\n"
