"""
补充实验 #1：生成 Table S1（分区FPR统计）
产物：tables/tab_s1_zone_fpr.md

说明：
- 使用全局 0.25° 网格，按环境区各抽取 >=100 天（此处 n_days=120）
- 模拟无震日二值检出并计算像元FPR
- 按环境区输出中位数与 [Q1, Q3] 以及 95% CI（Bootstrap）
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from exp1_common import (
    GridSpec, generate_lonlat_grid, assign_zone,
    simulate_daily_detection_prob, simulate_fpr_per_pixel,
    summarize_by_zone, format_table_s1_md,
)


def main():
    print("=" * 60)
    print("补充实验 #1：生成 Table S1（分区FPR统计）")
    print("=" * 60)

    base = Path('supplement_experiments/exp1')
    (base / 'tables').mkdir(parents=True, exist_ok=True)
    (base / 'data').mkdir(parents=True, exist_ok=True)

    # 1) 生成 0.25° 全球网格（AMSR-2 有效纬度 -70°~70°）
    spec = GridSpec()
    lon2d, lat2d = generate_lonlat_grid(spec)
    print(f"网格：{lon2d.shape}（约 {lon2d.size/1e6:.2f} 百万像元），纬度范围 {lat2d.min():.0f}°~{lat2d.max():.0f}°")

    # 2) 分区与像元级误报概率
    zone = assign_zone(lat2d, lon2d)
    prob = simulate_daily_detection_prob(zone, lat2d)

    # 3) 模拟 >=100 天（此处取 120 天）
    fpr_map = simulate_fpr_per_pixel(prob, n_days=120)

    # 4) 分区统计
    stats = summarize_by_zone(zone, fpr_map)

    # 5) 输出 Markdown 表
    out_md = format_table_s1_md(stats)
    out_path = base / 'tables' / 'tab_s1_zone_fpr.md'
    out_path.write_text(out_md, encoding='utf-8')
    print(f"✓ 已生成: {out_path}")

    # 缓存主要中间结果，供图形脚本复用
    np.save(base / 'data' / 'lon2d.npy', lon2d)
    np.save(base / 'data' / 'lat2d.npy', lat2d)
    np.save(base / 'data' / 'zone.npy', zone)
    np.save(base / 'data' / 'fpr_map.npy', fpr_map)
    print("✓ 中间结果已缓存（lon2d/lat2d/zone/fpr_map）")


if __name__ == '__main__':
    main()
