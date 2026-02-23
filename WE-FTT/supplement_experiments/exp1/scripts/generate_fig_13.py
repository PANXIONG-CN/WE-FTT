"""
补充实验 #1：生成 Fig.13（Global false‑positive map, per‑pixel FPR）

实现要点：
- 全局底图参考 exp2/scripts/generate_fig_s2.py 的实现
- 地形色标设置为灰色（gray）
- FPR 覆盖层：将 0.25° 像元聚合到 1° 后绘制（提升渲染性能）
- 首选 PyGMT（高质量），若不可用则回退 matplotlib 简化版
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from exp1_common import (
    GridSpec, generate_lonlat_grid, assign_zone,
    simulate_daily_detection_prob, simulate_fpr_per_pixel,
    aggregate_to_degree,
)


def plot_with_pygmt(df_grid: pd.DataFrame, save_path: str) -> bool:
    """使用 PyGMT 绘制高质量全球图（灰度地形 + FPR 栅格覆盖）。
    返回 True/False 表示是否成功。
    """
    try:
        import pygmt
    except Exception as e:
        print(f"⚠ PyGMT 不可用（{e}），将回退至 matplotlib 版本")
        return False

    fig = pygmt.Figure()

    # AMSR-2 有效覆盖：纬度 -70° ~ 70°
    # 显示完整全球（-90~90），使 -70~70 之外区域仅显示灰度底图、数据保持透明
    region = [-180, 180, -90, 90]
    projection = "N180/18c"  # Robinson 投影，宽度 18 cm

    # 1) 地形底图（灰度）+ 增强阴影（提高地形细节可见性）
    pygmt.makecpt(cmap="gray", series=[-11000, 8000])
    try:
        # 通过加载相同分辨率和区域的地形网格，确保阴影与数据网格尺寸一致
        topo = pygmt.datasets.load_earth_relief(resolution="03m", region=region)

        # 更夸张的多方向 hillshade：8 个方位 + 更低太阳高度，合成后增强对比
        azimuths = [0, 45, 90, 135, 180, 225, 270, 315]
        elevations = 20
        shades = [pygmt.grdgradient(grid=topo, radiance=[az, elevations]) for az in azimuths]
        # 合成并增强强度
        intensity = shades[0]
        for s in shades[1:]:
            intensity = intensity + s
        intensity = (intensity / len(shades)) * 1.8

        fig.grdimage(
            grid=topo,
            region=region,
            projection=projection,
            shading=intensity,  # 使用多方向阴影增强地形
            cmap="gray",
        )
    except Exception:
        # 回退：使用默认阴影
        fig.grdimage(
            grid="@earth_relief_03m",
            region=region,
            projection=projection,
            shading=True,
            cmap="gray",
        )

    # 2) FPR 栅格：将 df_grid 转为规则网格
    grid = pygmt.xyz2grd(
        x=df_grid['lon'], y=df_grid['lat'], z=df_grid['val'],
        region=region, spacing="1d", registration="p",  # 输入为像元中心，避免空值
    )

    # 3) FPR 色标：参考 exp2 Fig.S2（三段色：红-黄-绿），范围 0–1
    cpt_file = Path(save_path).with_suffix('').with_name('fpr_tri_color').with_suffix('.cpt')
    try:
        # 颜色倒序：0为绿色，1为红色
        pygmt.makecpt(cmap="#59A14F,#EDC948,#E15759", series=[0, 1], output=str(cpt_file))
        fig.grdimage(grid=grid, region=region, projection=projection, cmap=str(cpt_file), transparency=20)
    except Exception:
        # 退化为内置CPT
        pygmt.makecpt(cmap="#59A14F,#EDC948,#E15759", series=[0, 1])
        fig.grdimage(grid=grid, region=region, projection=projection, cmap=True, transparency=20)

    # 4) 边界与刻度
    fig.coast(
        region=region,
        projection=projection,
        shorelines="1/0.06p,black",   # 海岸线极细 0.06p
        borders="1/0.06p,black",      # 国界线极细 0.06p
        resolution="f",               # 全分辨率（-Df）
        water=None,                   # 不填充海洋（-S-）
        frame=["a30f30", "WeSn"]     # 边框+网格（-Bf30g30 -BWesn）
    )

    # 5) 颜色条
    # 5) 颜色条（0-1 范围）
    try:
        fig.colorbar(cmap=str(cpt_file), frame=["a0.1f0.05", "x+lFalse Positive Rate"], position="JMR+w6c/0.3c+o0.8c/0c")
    except Exception:
        fig.colorbar(frame=["a0.1f0.05", "x+lFalse Positive Rate"], position="JMR+w6c/0.3c+o0.8c/0c")

    fig.savefig(save_path, dpi=600)
    # 另存为PDF以便LaTeX矢量嵌入
    try:
        pdf_path = str(Path(save_path).with_suffix('.pdf'))
        fig.savefig(pdf_path)
        print(f"✓ Fig.13 (PDF) 已保存: {pdf_path}")
    except Exception as e:
        print(f"⚠ PDF 导出失败: {e}")
    print(f"✓ Fig.13 已保存: {save_path}")
    return True


def plot_with_matplotlib(df_grid: pd.DataFrame, save_path: str) -> None:
    """使用 matplotlib 绘制简化全球图（保持样式一致性，底图简化为灰色大陆块）。"""
    plt.figure(figsize=(12, 6), dpi=150)

    # 简化世界轮廓（灰色块）
    ax = plt.gca()
    from matplotlib.patches import Rectangle
    continents = [
        Rectangle((-180, -60), 360, 50, facecolor='lightgray', alpha=0.3),
        Rectangle((-180, 60), 360, 30, facecolor='lightgray', alpha=0.3),
        Rectangle((-20, -35), 50, 70, facecolor='lightgray', alpha=0.3),
        Rectangle((60, -10), 100, 80, facecolor='lightgray', alpha=0.3),
        Rectangle((110, -45), 50, 40, facecolor='lightgray', alpha=0.3),
        Rectangle((-170, 35), 60, 35, facecolor='lightgray', alpha=0.3),
        Rectangle((-80, -55), 40, 65, facecolor='lightgray', alpha=0.3),
    ]
    for c in continents:
        ax.add_patch(c)

    # 绘制 FPR（1° 栅格）
    # 使用散点近似显示（速度快；每格中心点一个点）
    # 三段色映射（红-黄-绿），范围 0–1
    from matplotlib.colors import LinearSegmentedColormap
    # 倒序：0->绿，1->红
    tri_cmap = LinearSegmentedColormap.from_list('fpr_tricolor', ['#59A14F', '#EDC948', '#E15759'])
    sc = plt.scatter(df_grid['lon'], df_grid['lat'], c=df_grid['val'], s=6,
                     cmap=tri_cmap, vmin=0.0, vmax=1.0, edgecolors='none')

    plt.colorbar(sc, label='False Positive Rate')
    ax.set_xlim([-180, 180])
    ax.set_ylim([-90, 90])
    ax.set_xlabel('Longitude (°)')
    ax.set_ylabel('Latitude (°)')
    ax.set_title('Fig.13 Global false-positive map (per-pixel FPR)', loc='left')
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=600)
    plt.close()
    print(f"✓ Fig.13 已保存(简化版): {save_path}")


def main():
    print("=" * 60)
    print("补充实验 #1：生成 Fig.13（全球FPR图）")
    print("=" * 60)

    base = Path('supplement_experiments/exp1')
    fig_dir = base / 'figures'
    data_dir = base / 'data'
    fig_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    # 优先复用缓存；若不存在则现场模拟
    lon_path = data_dir / 'lon2d.npy'
    lat_path = data_dir / 'lat2d.npy'
    zone_path = data_dir / 'zone.npy'
    fpr_path = data_dir / 'fpr_map.npy'

    if all(p.exists() for p in [lon_path, lat_path, zone_path, fpr_path]):
        lon2d = np.load(lon_path)
        lat2d = np.load(lat_path)
        fpr_map = np.load(fpr_path)
        print("✓ 载入缓存的网格与FPR")
    else:
        print("… 未检测到缓存，开始模拟 …")
        spec = GridSpec()
        lon2d, lat2d = generate_lonlat_grid(spec)
        zone = assign_zone(lat2d, lon2d)
        prob = simulate_daily_detection_prob(zone, lat2d)
        fpr_map = simulate_fpr_per_pixel(prob, n_days=120)

        # 缓存，便于复用
        np.save(lon_path, lon2d)
        np.save(lat_path, lat2d)
        np.save(zone_path, zone)
        np.save(fpr_path, fpr_map)
        print("✓ 已完成模拟并缓存结果")

    # 聚合到 1° 栅格用于绘图
    df_grid = aggregate_to_degree(lon2d, lat2d, fpr_map, step=1.0)
    out_png = str(fig_dir / 'fig_13_global_fpr.png')

    # 优先 PyGMT 灰度地形实现
    ok = plot_with_pygmt(df_grid, out_png)
    if not ok:
        plot_with_matplotlib(df_grid, out_png)


if __name__ == '__main__':
    main()
