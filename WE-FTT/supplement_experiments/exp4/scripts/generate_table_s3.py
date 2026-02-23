#!/usr/bin/env python3
"""
生成 Tab S3：分层（Magnitude × Depth）MCC/FPR 与 95% CI + Δ（相对全体均值）。

输入：data/raw/earthquake_catalog.csv
输出：
  - tables/table_s3_strata_performance.csv
  - tables/table_s3_strata_performance.tex
  - tables/summary_exp4.json
"""

from pathlib import Path
import sys
sys.path.append('.')

import pandas as pd

from supplement_experiments.exp4.scripts.exp4_common import (
    SimulationConfig, read_catalog_counts, simulate_strata_metrics,
    export_table_tex_csv, save_json_summary
)


def main():
    project_root = Path('.')
    catalog_path = project_root / 'data' / 'raw' / 'earthquake_catalog.csv'
    exp4_dir = project_root / 'supplement_experiments' / 'exp4'
    tables_dir = exp4_dir / 'tables'

    cfg = SimulationConfig()

    # 1) 读取分层计数
    counts_df = read_catalog_counts(catalog_path)

    # 2) 模拟与统计
    results = simulate_strata_metrics(counts_df, cfg)
    per = results['per_stratum']

    # 3) 导出 CSV 与 LaTeX
    csv_path, tex_path = export_table_tex_csv(per, tables_dir)

    # 4) 保存简要摘要（供调试复用）
    save_json_summary(results, tables_dir / 'summary_exp4.json')

    print(f"✓ 已生成: {csv_path}")
    print(f"✓ 已生成: {tex_path}")


if __name__ == '__main__':
    main()

