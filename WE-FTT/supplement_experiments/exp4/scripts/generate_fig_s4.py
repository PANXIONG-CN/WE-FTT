#!/usr/bin/env python3
"""
生成 Fig S4：分层（Magnitude × Depth）MCC/FPR 森林图（双子图）。

输入：data/raw/earthquake_catalog.csv
输出：figures/fig_s4_forest_by_strata.{pdf,png,svg}
"""

from pathlib import Path
import sys
sys.path.append('.')

import pandas as pd

from supplement_experiments.exp4.scripts.exp4_common import (
    SimulationConfig, read_catalog_counts, simulate_strata_metrics, render_forest_plot
)


def main():
    project_root = Path('.')
    catalog_path = project_root / 'data' / 'raw' / 'earthquake_catalog.csv'
    exp4_dir = project_root / 'supplement_experiments' / 'exp4'
    figures_dir = exp4_dir / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)

    cfg = SimulationConfig()

    counts_df = read_catalog_counts(catalog_path)
    results = simulate_strata_metrics(counts_df, cfg)
    per = results['per_stratum']
    overall = results['overall']

    save_prefix = figures_dir / 'fig_s4_forest_by_strata'
    render_forest_plot(per, overall, save_prefix)
    print(f"✓ 已生成: {save_prefix}.pdf/.png/.svg")


if __name__ == '__main__':
    main()

