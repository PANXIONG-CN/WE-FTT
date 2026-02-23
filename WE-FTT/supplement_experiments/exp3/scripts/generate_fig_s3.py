"""
生成 Fig. S3：海域条件检出对比条形图
流程：
1) 读取 data/raw/earthquake_catalog.csv，筛选海域 M≥7.0 的10–20个候选
2) 调用 NOAA/NCEI hazard-service 检索海啸到达（缓存）
3) 按三类控制生成指标：检出率、FPR、MCC；并统计显著性
4) 绘图（PDF/PNG/SVG）与题注（中英）
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('.')

try:
    from supplement_experiments.nature_style import setup_nature_style, create_colorbar, save_figure
    from supplement_experiments.utils import PAPER_COLORS
except Exception:
    from nature_style import setup_nature_style, create_colorbar, save_figure  # type: ignore
    from utils import PAPER_COLORS  # type: ignore

from exp3_common import (
    read_and_select_marine_events,
    fetch_tsunami_arrival_noaa,
    apply_marine_controls,
    write_candidates_and_cache,
    write_markdown_outputs,
)


def plot_bars(results: dict, save_prefix: str) -> None:
    # 套用Nature风格并统一字体（参考exp2 Fig S1）
    setup_nature_style()
    plt.rcParams['font.family'] = ['serif']
    plt.rcParams['font.serif'] = [
        'DejaVu Serif', 'Bitstream Vera Serif', 'Computer Modern Roman',
        'New Century Schoolbook', 'Century Schoolbook L', 'Utopia', 'ITC Bookman',
        'Bookman', 'Nimbus Roman No9 L', 'Times New Roman', 'Times', 'Palatino',
        'Charter', 'serif'
    ]
    plt.rcParams['mathtext.fontset'] = 'stix'

    conditions = ['Baseline', 'Pre-EQ Only', 'No Tsunami', 'No Wind', 'Both']
    keys = ['baseline', 'pre_earthquake_only', 'tsunami_excluded', 'wind_waves_excluded', 'both_excluded']

    det = [results[k]['detection_rate'] for k in keys]
    fpr = [results[k]['fpr'] for k in keys]

    colors = ['#2E4053', '#5DADE2', '#F39C12', '#E74C3C', '#A569BD']

    # A4尺寸，上下排版
    fig = plt.figure(figsize=(8.27, 11.69))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.22,
                          top=0.92, bottom=0.08, left=0.12, right=0.88)

    # 子图(a)：检出率
    ax1 = fig.add_subplot(gs[0])
    ax1.text(0.02, 0.98, '(a)', transform=ax1.transAxes,
             fontsize=12, fontweight='bold', va='top', ha='left')

    b1 = ax1.bar(range(len(conditions)), det, color=colors, alpha=0.8,
                 edgecolor='black', linewidth=0.8)

    # 随机水平线与基线参考线
    ax1.axhline(0.5, color='red', linestyle='--', linewidth=1.2, alpha=0.6, label='Random (0.5)')
    ax1.axhline(results['baseline']['detection_rate'], color='blue', linestyle=':',
                linewidth=1.5, alpha=0.7, label='Baseline')

    ax1.set_xticks(range(len(conditions)))
    ax1.set_xticklabels(conditions, rotation=20)
    ax1.set_ylabel('Detection Rate', fontsize=9)
    # 聚焦显示范围以增强对比：固定到 0.40–0.60
    ax1.set_ylim(0.40, 0.60)
    # 避免数值标签越界
    y_max = ax1.get_ylim()[1]
    for i, v in enumerate(det):
        label_y = min(v + 0.010, y_max - 0.005)
        ax1.text(i, label_y, f"{v:.3f}", ha='center', va='bottom', fontsize=8)
    ax1.legend(loc='upper right', fontsize=8, frameon=True, framealpha=0.9)
    ax1.grid(True, alpha=0.2, axis='y', linewidth=0.5)

    # 显著性标记（相对基线）
    stats = results.get('statistics', {})
    for i, cond in enumerate(['pre_only', 'no_tsunami', 'no_wind', 'both_filtered']):
        res = stats.get(cond, {})
        if res.get('significant', False):
            ax1.text(i + 1, det[i + 1] + 0.03, '*', ha='center', fontsize=13, color='black')

    # 子图(b)：FPR
    ax2 = fig.add_subplot(gs[1])
    ax2.text(0.02, 0.98, '(b)', transform=ax2.transAxes,
             fontsize=12, fontweight='bold', va='top', ha='left')

    b2 = ax2.bar(range(len(conditions)), fpr, color=colors, alpha=0.8,
                 edgecolor='black', linewidth=0.8)
    for i, v in enumerate(fpr):
        ax2.text(i, v + 0.002, f"{v:.3f}", ha='center', va='bottom', fontsize=8)
    ax2.set_xticks(range(len(conditions)))
    ax2.set_xticklabels(conditions, rotation=20)
    ax2.set_ylabel('False Positive Rate', fontsize=9)
    # 固定到 0.00–0.06
    ax2.set_ylim(0.00, 0.06)
    ax2.grid(True, alpha=0.2, axis='y', linewidth=0.5)
    # 避免右图数值标签越界
    y2_max = ax2.get_ylim()[1]
    for i, v in enumerate(fpr):
        label_y = min(v + 0.002, y2_max - 0.001)
        ax2.text(i, label_y, f"{v:.3f}", ha='center', va='bottom', fontsize=8)

    # 子图(a)左上角插入统计信息（vs random 的 p-values），避免与(a)重叠，向下错位
    label_map = {
        'baseline_vs_random': 'Baseline',
        'pre_only_vs_random': 'Pre-EQ Only',
        'no_tsunami_vs_random': 'No Tsunami',
        'no_wind_vs_random': 'No Wind',
        'both_filtered_vs_random': 'Both Filtered',
    }
    # 使用逐行拼接，避免末尾多余空行
    lines = ['Statistical Tests (p-values):']
    for key in ['baseline_vs_random', 'pre_only_vs_random', 'no_tsunami_vs_random', 'no_wind_vs_random', 'both_filtered_vs_random']:
        p = stats.get(key, {}).get('p_value', None)
        if p is not None:
            lines.append(f"{label_map[key]}: p={p:.2e}")
    stats_text = '\n'.join(lines)
    ax1.text(0.02, 0.92, stats_text, transform=ax1.transAxes,
             va='top', ha='left', fontsize=8,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='gray'))

    save_figure(fig, save_prefix, formats=['pdf', 'png', 'svg'], dpi=600)
    plt.close(fig)


def main():
    print('=' * 60)
    print('生成 Fig. S3：海啸与极端风浪伪影敏感性分析')
    print('=' * 60)

    root = Path('supplement_experiments/exp3')
    data_dir = root / 'data'
    fig_dir = root / 'figures'
    docs_dir = root / 'docs'
    tables_dir = root / 'tables'
    for d in [data_dir, fig_dir, docs_dir, tables_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 1) 读取与筛选
    csv_path = Path('data/raw/earthquake_catalog.csv')
    events = read_and_select_marine_events(str(csv_path), max_events=20)
    if events.empty or len(events) < 1:
        print('警告：未筛选到M≥7.0海域事件，后续统计可能无效。')

    # 2) 海啸到达检索（带缓存）
    cache_json = data_dir / 'tsunami_events.json'
    arrivals = fetch_tsunami_arrival_noaa(events, str(cache_json))

    # 写出候选与缓存
    candidates_csv = data_dir / 'earthquake_candidates.csv'
    write_candidates_and_cache(events, str(candidates_csv), arrivals, str(cache_json))

    # 3) 控制实验
    results = apply_marine_controls(events, arrivals, mcc_baseline=0.84)

    # 4) 作图与文稿
    save_prefix = str(fig_dir / 'fig_s3_marine_controls')
    plot_bars(results, save_prefix)

    # 题注与表格
    write_markdown_outputs(results,
                           fig_caption_path=str(docs_dir / 'FIG_S3_CAPTION.md'),
                           table_path=str(tables_dir / 'table_s3_marine_controls.md'))

    print('完成：')
    print(f'- 候选事件: {candidates_csv}')
    print(f'- 海啸缓存: {cache_json}')
    print(f'- 图件: {save_prefix}.pdf / .png / .svg')
    print(f'- 题注: {docs_dir / "FIG_S3_CAPTION.md"}')
    print(f'- 表格: {tables_dir / "table_s3_marine_controls.md"}')


if __name__ == '__main__':
    main()
