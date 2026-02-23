"""
生成 Table S3：Marine Zone Controls 指标表
可独立运行，复用与生成图相同的数据流程。
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.append('.')

from exp3_common import (
    read_and_select_marine_events,
    fetch_tsunami_arrival_noaa,
    apply_marine_controls,
    write_candidates_and_cache,
    write_markdown_outputs,
)


def main():
    root = Path('supplement_experiments/exp3')
    data_dir = root / 'data'
    tables_dir = root / 'tables'
    docs_dir = root / 'docs'
    for d in [data_dir, tables_dir, docs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    csv_path = Path('data/raw/earthquake_catalog.csv')
    events = read_and_select_marine_events(str(csv_path), max_events=20)

    cache_json = data_dir / 'tsunami_events.json'
    arrivals = fetch_tsunami_arrival_noaa(events, str(cache_json))
    write_candidates_and_cache(events, str(data_dir / 'earthquake_candidates.csv'), arrivals, str(cache_json))

    results = apply_marine_controls(events, arrivals, mcc_baseline=0.84)

    table_path = tables_dir / 'table_s3_marine_controls.md'
    caption_path = docs_dir / 'FIG_S3_CAPTION.md'
    write_markdown_outputs(results, fig_caption_path=str(caption_path), table_path=str(table_path))

    print(f'✓ 表格已生成: {table_path}')


if __name__ == '__main__':
    main()

