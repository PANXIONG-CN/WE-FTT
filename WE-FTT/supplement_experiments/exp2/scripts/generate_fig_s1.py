"""
补充实验 #2：生成Figure S1（时间演化图）
时间性能时间线和区域平均性能条形图
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import sys
sys.path.append('.')

# 导入共享函数
from exp2_common import fetch_usgs_earthquakes, evaluate_out_of_sample_performance

# 导入工具函数
import sys
from pathlib import Path
# 添加supplement_experiments到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
try:
    from supplement_experiments.nature_style import *
    from supplement_experiments.utils import PAPER_COLORS, ZONE_DEFINITIONS
except:
    from nature_style import *
    from utils import PAPER_COLORS, ZONE_DEFINITIONS

def plot_temporal_evolution_nature(results: dict, save_path: str = None):
    """
    绘制Fig. S1：事件时间序列图（Nature风格）

    Args:
        results: 评估结果
        save_path: 保存路径
    """
    # 设置Nature风格
    setup_nature_style()

    # 设置全局字体为serif
    plt.rcParams['font.family'] = ['serif']
    plt.rcParams['font.serif'] = ['DejaVu Serif', 'Bitstream Vera Serif',
                                    'Computer Modern Roman', 'New Century Schoolbook',
                                    'Century Schoolbook L', 'Utopia', 'ITC Bookman',
                                    'Bookman', 'Nimbus Roman No9 L', 'Times New Roman',
                                    'Times', 'Palatino', 'Charter', 'serif']
    plt.rcParams['mathtext.fontset'] = 'stix'

    # 创建A4大小图表 (210mm × 297mm = 8.27" × 11.69")
    fig = plt.figure(figsize=(8.27, 11.69))
    # 调整子图间距
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.18,
                          top=0.92, bottom=0.08, left=0.12, right=0.88)

    events_df = results['events']
    perf_df = pd.DataFrame(results['event_performance'])

    # ========== 子图1：时间线和性能 ==========
    ax1 = fig.add_subplot(gs[0])

    # 添加子图编号(a)
    ax1.text(0.02, 0.98, '(a)', transform=ax1.transAxes,
            fontsize=12, fontweight='bold', va='top', ha='left')

    dates = pd.to_datetime(events_df['date'])
    magnitudes = events_df['magnitude'].values
    mccs = perf_df['MCC'].values

    # 创建颜色映射（基于MCC值）
    from matplotlib.colors import LinearSegmentedColormap
    colors_list = ['#E15759', '#EDC948', '#59A14F']  # 红-黄-绿
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('mcc', colors_list, N=n_bins)
    norm = plt.Normalize(vmin=0.75, vmax=0.82)

    # 绘制每个事件
    label_positions = []  # 存储已放置标签的位置，避免重叠
    for i, (date, mag, mcc) in enumerate(zip(dates, magnitudes, mccs)):
        color = cmap(norm(mcc))

        # 垂直线
        ax1.vlines(date, 6.8, mag, colors=color, linewidth=1.5, alpha=0.7)

        # 圆圈表示震级
        size = 50 * (mag - 6.5)
        ax1.scatter(date, mag, s=size, c=[color], edgecolor='black',
                   linewidth=0.8, alpha=0.85, zorder=5)

        # 震中标签：显示震级与MCC，自动堆叠避免重叠
        label_text = f"M{mag:.1f}\nMCC {mcc:.2f}"
        base_offset = 0.12
        label_y = mag + base_offset
        date_num = mdates.date2num(date)
        for existing_x, existing_y in label_positions:
            if abs(date_num - existing_x) < 25 and label_y <= existing_y + 0.12:
                label_y = existing_y + 0.12
        label_positions.append((date_num, label_y))
        ax1.text(date, label_y, label_text,
                ha='center', va='bottom', fontsize=6,
                fontweight='normal', linespacing=1.05)

    ax1.set_ylabel('Magnitude', fontsize=9)
    ax1.set_ylim([6.8, 9.0])
    ax1.set_xlim([dates.min() - pd.Timedelta(days=30),
                  dates.max() + pd.Timedelta(days=30)])
    ax1.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)

    # 添加颜色条
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = create_colorbar(fig, ax1, sm, label='MCC',
                          location='right', width=0.015, pad=0.02)

    # 格式化x轴日期
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha='right')

    # ========== 子图2：区域性能条形图 ==========
    ax2 = fig.add_subplot(gs[1])

    # 添加子图编号(b)
    ax2.text(0.02, 0.98, '(b)', transform=ax2.transAxes,
            fontsize=12, fontweight='bold', va='top', ha='left')

    # 按区域分组
    zone_performance = perf_df.groupby('zone')['MCC'].agg(['mean', 'std', 'count'])
    zones = zone_performance.index.tolist()

    # 使用论文配色
    zone_colors = [PAPER_COLORS[f'zone_{z.lower()}'] for z in zones]

    x_pos = np.arange(len(zones))
    bars = ax2.bar(x_pos, zone_performance['mean'].values,
                   color=zone_colors, alpha=0.7,
                   edgecolor='black', linewidth=0.8)

    # 添加误差条
    ax2.errorbar(x_pos, zone_performance['mean'].values,
                yerr=zone_performance['std'].values,
                fmt='none', ecolor='black', capsize=3, linewidth=1)

    # 添加基线
    ax2.axhline(y=results['comparison']['in_sample_mcc'],
               color='red', linestyle='--', linewidth=1.5,
               alpha=0.6, label='In-sample baseline')
    ax2.axhline(y=results['overall_metrics']['MCC'],
               color='blue', linestyle='-', linewidth=1.5,
               alpha=0.6, label='Out-of-sample mean')

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f'Zone {z}\n({ZONE_DEFINITIONS[z]["name"]})'
                        for z in zones], fontsize=7)
    ax2.set_ylabel('Average MCC', fontsize=9)
    ax2.set_ylim([0.70, 0.85])
    ax2.legend(loc='upper right', fontsize=7, frameon=True,
              edgecolor='gray', framealpha=0.9)
    ax2.grid(True, alpha=0.2, axis='y', linewidth=0.5)

    # 添加样本数量标注
    for i, (zone, row) in enumerate(zone_performance.iterrows()):
        ax2.text(i, 0.705, f'n={int(row["count"])}',
                ha='center', va='bottom', fontsize=6)

    # 不使用tight_layout，保持手动设置的间距

    if save_path:
        save_figure(fig, save_path.replace('.pdf', ''),
                   formats=['pdf', 'png', 'svg'], dpi=600)
        print(f"✓ Figure S1已保存至: {save_path}")

    return fig

def main():
    """
    主函数：生成Figure S1
    """
    print("=" * 60)
    print("生成Figure S1：时间演化图")
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

    # 3. 生成Figure S1
    print("\n步骤3: 生成Figure S1")
    fig1_path = fig_dir / 'fig_s1_temporal_evolution.pdf'
    fig1 = plot_temporal_evolution_nature(results, save_path=str(fig1_path))
    plt.close()

    print("\n" + "=" * 60)
    print("✓ Figure S1生成完成！")
    print("=" * 60)
    print(f"\n输出文件:")
    print(f"  - {fig_dir / 'fig_s1_temporal_evolution.pdf'}")
    print(f"  - {fig_dir / 'fig_s1_temporal_evolution.png'}")
    print(f"  - {fig_dir / 'fig_s1_temporal_evolution.svg'}")

    return results

if __name__ == "__main__":
    main()
