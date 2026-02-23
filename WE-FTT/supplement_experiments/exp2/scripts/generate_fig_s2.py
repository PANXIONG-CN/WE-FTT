"""
补充实验 #2：生成Figure S2（空间分布图）
全球地震事件空间覆盖和Dobrovolsky半径图
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from pathlib import Path
import sys
sys.path.append('.')

KM_PER_DEGREE = 111.32  # 地理坐标下1°对应的近似公里数

# 导入共享函数
from exp2_common import fetch_usgs_earthquakes, evaluate_out_of_sample_performance

# 导入工具函数
import sys
from pathlib import Path
# 添加supplement_experiments到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
try:
    from supplement_experiments.nature_style import *
except:
    from nature_style import *

def plot_spatial_coverage_matplotlib(results: dict, save_path: str = None):
    """
    使用matplotlib绘制空间分布图

    Args:
        results: 评估结果
        save_path: 保存路径
    """
    setup_nature_style()

    # 设置全局字体为serif
    plt.rcParams['font.family'] = ['serif']
    plt.rcParams['font.serif'] = ['DejaVu Serif', 'Bitstream Vera Serif',
                                    'Computer Modern Roman', 'New Century Schoolbook',
                                    'Century Schoolbook L', 'Utopia', 'ITC Bookman',
                                    'Bookman', 'Nimbus Roman No9 L', 'Times New Roman',
                                    'Times', 'Palatino', 'Charter', 'serif']
    plt.rcParams['mathtext.fontset'] = 'stix'

    fig = create_figure('double', height_ratio=0.6)
    ax = fig.add_subplot(111)

    # 添加子图编号
    ax.text(0.02, 0.98, '(a)', transform=ax.transAxes,
            fontsize=12, fontweight='bold', va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                     edgecolor='black', linewidth=1))

    events_df = results['events']
    perf_df = pd.DataFrame(results['event_performance'])

    # 简化的世界地图轮廓
    continents = [
        Rectangle((-180, -60), 360, 50, facecolor='lightgray', alpha=0.3),
        Rectangle((-180, 60), 360, 30, facecolor='lightgray', alpha=0.3),
        Rectangle((-20, -35), 50, 70, facecolor='lightgray', alpha=0.3),
        Rectangle((60, -10), 100, 80, facecolor='lightgray', alpha=0.3),
        Rectangle((110, -45), 50, 40, facecolor='lightgray', alpha=0.3),
        Rectangle((-170, 35), 60, 35, facecolor='lightgray', alpha=0.3),
        Rectangle((-80, -55), 40, 65, facecolor='lightgray', alpha=0.3),
    ]
    for continent in continents:
        ax.add_patch(continent)

    # 颜色映射
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list('mcc', ['#E15759', '#EDC948', '#59A14F'])
    norm = plt.Normalize(vmin=0.75, vmax=0.82)

    # 绘制地震事件
    for idx, event in events_df.iterrows():
        mcc = perf_df.iloc[idx]['MCC']
        color = cmap(norm(mcc))

        # Dobrovolsky半径
        circle = Circle((event['longitude'], event['latitude']),
                       radius=event['radius_km']/111,
                       fill=False, edgecolor='#C62828',
                       linewidth=0.9, alpha=0.5, linestyle='--')
        ax.add_patch(circle)

        # 震中
        size = 18 * max(event['magnitude'] - 6.5, 0.3)
        ax.scatter(event['longitude'], event['latitude'],
                  s=size, c=[color], marker='o',
                  edgecolor='black', linewidth=0.8, alpha=0.85, zorder=5)

        # 标签：使用 年-月-日 和 震级
        date_str = pd.to_datetime(event['date']).strftime('%Y-%m-%d')
        mag_str = f"M{event['magnitude']:.1f}"
        ax.annotate(f"{date_str}\n{mag_str}",
                   xy=(event['longitude'], event['latitude']),
                   xytext=(3, 3), textcoords='offset points',
                   fontsize=5, ha='left')

    ax.set_xlim([-180, 180])
    ax.set_ylim([-90, 90])
    ax.set_xlabel('Longitude (°)', fontsize=9)
    ax.set_ylabel('Latitude (°)', fontsize=9)
    ax.set_title('Spatial distribution of out-of-sample events\n'
                'Event locations and Dobrovolsky preparation zones',
                fontsize=10, pad=10, loc='left')
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)

    # 颜色条
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = create_colorbar(fig, ax, sm, label='MCC',
                          location='right', width=0.015, pad=0.02)
    # 图例（颜色条）样式：去除黑色边框和端点三角
    try:
        cbar.set_extend('neither')
        cbar.outline.set_visible(False)
        for spine in cbar.ax.spines.values():
            spine.set_visible(False)
    except Exception:
        pass

    # 统计信息
    stats_text = (f"Total Events: {len(events_df)}\n"
                 f"Mean MCC: {results['overall_metrics']['MCC']:.3f}\n"
                 f"In-sample: {results['comparison']['in_sample_mcc']:.3f}\n"
                 f"Difference: {results['comparison']['difference']:.3f}")
    ax.text(0.02, 0.85, stats_text, transform=ax.transAxes,
           verticalalignment='top', horizontalalignment='left',
           bbox=dict(boxstyle='round', facecolor='white',
                    alpha=0.8, edgecolor='gray', linewidth=0.8),
           fontsize=7)

    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path.replace('.pdf', ''),
                   formats=['pdf', 'png', 'svg'], dpi=600)
        print(f"✓ Figure S2已保存至: {save_path}")

    return fig

def plot_spatial_coverage_pygmt(results: dict, save_path: str = None):
    """
    绘制Fig. S2：全球地震分布图（PyGMT高质量版本）
    使用地形底图和Robinson投影，600 DPI输出

    Args:
        results: 评估结果
        save_path: 保存路径
    """
    try:
        import pygmt
    except (ImportError, Exception) as e:
        print(f"⚠ PyGMT不可用 ({e})，使用matplotlib绘制简化版地图")
        print("提示：如需启用PyGMT高质量渲染，请使用包装脚本运行：")
        print("  ./run_with_gmt.sh supplement_experiments/exp2/scripts/generate_fig_s2.py")
        return plot_spatial_coverage_matplotlib(results, save_path)

    events_df = results['events']
    perf_df = pd.DataFrame(results['event_performance'])

    # 创建图表
    fig = pygmt.Figure()

    # 1. 设置全局区域和投影
    region = [-180, 180, -90, 90]  # 全球范围
    projection = "N180/18c"  # Robinson投影，中心经度180°，宽度18cm

    # 2. 创建地形色标（etopo1，范围-11000到0）
    pygmt.makecpt(
        cmap="etopo1",
        series=[-11000, 0],
    )

    # 3. 绘制地形底图（@earth_relief_03m + 地形阴影）
    fig.grdimage(
        grid="@earth_relief_03m",
        region=region,
        projection=projection,
        shading=True  # 对应GMT的 -I 参数
    )

    # 4. 添加海岸线和国界（对应GMT coast命令）
    fig.coast(
        region=region,
        projection=projection,
        borders="1/0.1p,black",      # 国界线（-N1/0.1p,black）
        resolution="f",               # 全分辨率（-Df）
        water=None,                   # 不填充海洋（-S-）
        frame=["a30f30", "WeSn"]     # 边框+网格（-Bf30g30 -BWesn）
    )

    # 5. 创建MCC颜色映射CPT文件（与fig_s1相同：红-黄-绿，0.75-0.82）
    cpt_file = Path(save_path).parent / 'mcc_earthquakes.cpt' if save_path else 'mcc_earthquakes.cpt'
    pygmt.makecpt(
        cmap="#E15759,#EDC948,#59A14F",  # 红-黄-绿渐变
        series=[0.75, 0.82],
        output=str(cpt_file)
    )

    # 6. 逐个绘制地震震中（确保颜色正确映射）
    for idx, event in events_df.iterrows():
        mcc = perf_df.iloc[idx]['MCC']
        mag = event['magnitude']
        lon = event['longitude']
        lat = event['latitude']

        # 圆圈大小（cm）
        size = max(0.12, 0.15 + (mag - 7.0) * 0.07)

        # 绘制单个地震点（使用fill参数代替color）
        fig.plot(
            x=lon,
            y=lat,
            region=region,
            projection=projection,
            style=f"c{size}c",  # 圆圈，指定大小
            cmap=str(cpt_file),  # 显式指定MCC色标
            fill="+z",  # 通过-Z和CPT执行填充
            zvalue=mcc,  # 将MCC值映射到CPT
            pen="0.8p,black"  # 黑色边框
        )

    # 7. 绘制Dobrovolsky半径圆圈
    for idx, event in events_df.iterrows():
        radius_deg = event['radius_km'] / KM_PER_DEGREE  # km转度（近似）

        # 使用PyGMT绘制圆圈
        fig.plot(
            x=event['longitude'],
            y=event['latitude'],
            region=region,
            projection=projection,
            style=f"E-{radius_deg*2}d",  # 椭圆直径（度数单位）
            pen="0.6p,#C62828,dashed",
            transparency=60
        )

    # 8. 智能标签放置（避免重叠）
    label_positions = []
    for idx, event in events_df.iterrows():
        mcc = perf_df.iloc[idx]['MCC']
        lon = event['longitude']
        lat = event['latitude']

        # 检查附近是否有其他地震
        nearby_count = sum(
            1 for idx2, other in events_df.iterrows()
            if idx != idx2 and
            abs(lon - other['longitude']) < 15 and
            abs(lat - other['latitude']) < 15
        )

        # 根据附近地震数量调整标签位置
        if nearby_count == 0:
            offset_lat = -4  # 默认下方
            justify = "CT"
        elif nearby_count == 1:
            offset_lat = 4  # 上方
            justify = "CB"
        elif nearby_count >= 2:
            offset_lat = -6  # 更远下方
            justify = "CT"

        # 标签文本：使用 年-月-日 和 震级
        date_str = pd.to_datetime(event['date']).strftime('%Y-%m-%d')
        mag_str = f"M{event['magnitude']:.1f}"
        label_positions.append({
            'lon': lon,
            'lat': lat + offset_lat,
            'text': f"{date_str}\\n{mag_str}",
            'justify': justify
        })

    # 9. 添加地震标签（带半透明背景）
    for label in label_positions:
        fig.text(
            x=label['lon'],
            y=label['lat'],
            text=label['text'],
            region=region,
            projection=projection,
            font="5p,Helvetica,black",
            justify=label['justify'],
            fill="white@30",  # 半透明白色背景
            clearance="1p/1p",
            pen="0.3p,gray"
        )

    # 10. 添加MCC颜色条
    fig.colorbar(
        cmap=str(cpt_file),
        frame=["a0.01f0.005", "x+lMCC"],
        position="JMR+w6c/0.3c+o0.8c/0c"
    )

    # 11. 保存为PNG（600 DPI）
    if save_path:
        fig.savefig(save_path, dpi=600, transparent=False)
        print(f"✓ Figure S2已保存至: {save_path}")

        # 清理临时CPT文件
        try:
            Path(cpt_file).unlink(missing_ok=True)
        except:
            pass

    return fig

def main():
    """
    主函数：生成Figure S2（PNG格式，600 DPI）
    """
    print("=" * 60)
    print("生成Figure S2：空间分布图（PyGMT专业版）")
    print("=" * 60)

    # 创建输出目录
    output_dir = Path('supplement_experiments/exp2')
    fig_dir = output_dir / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)

    data_dir = output_dir / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)

    # 1. 获取USGS真实地震数据
    print("\n步骤1: 获取USGS地震数据")
    data_path = data_dir / 'usgs_earthquakes_2023-2025.json'
    events_df = fetch_usgs_earthquakes(
        starttime='2023-08-01',
        endtime='2025-09-30',
        minmag=7.0,
        save_path=str(data_path)
    )

    if events_df.empty:
        print("❌ 无法获取地震数据，退出。")
        return None

    print(f"✓ 成功获取 {len(events_df)} 个地震事件")

    # 2. 评估性能
    print("\n步骤2: 评估样本外性能")
    results = evaluate_out_of_sample_performance(events_df, base_mcc=0.84)

    # 3. 生成Figure S2（仅PNG格式，600 DPI）
    print("\n步骤3: 生成Figure S2")
    fig2_path = fig_dir / 'fig_s2_spatial_coverage.png'

    # 使用PyGMT绘制专业版地图
    fig2 = plot_spatial_coverage_pygmt(results, save_path=str(fig2_path))

    if fig2 is None:
        print("❌ 图表生成失败")
        return None

    print("\n" + "=" * 60)
    print("✓ Figure S2生成完成！")
    print("=" * 60)
    print(f"\n输出文件:")
    print(f"  - {fig2_path}")

    return results

if __name__ == "__main__":
    main()
