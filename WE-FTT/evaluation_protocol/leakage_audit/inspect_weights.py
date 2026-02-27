#!/usr/bin/env python3
"""
快速检查 parquet 中的权重列是否为“预写入/静态列”，并输出可审计摘要（v1）。

用法示例：

python WE-FTT/evaluation_protocol/leakage_audit/inspect_weights.py \
  --parquet_path "/mnt/hdd_4tb_data/ArchivedWorks/MBT/FTT/updated_code/training_dataset.parquet" \
  --out_json "WE-FTT/evaluation_protocol/leakage_audit/traces/training_dataset_weights_snapshot.json"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# 将 WE-FTT 根目录加入路径（保证可直接以脚本方式运行）
weftt_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if weftt_root not in sys.path:
    sys.path.insert(0, weftt_root)

from evaluation_protocol.common.logging_utils import setup_logging  # noqa: E402


def _require_pyarrow():
    try:
        import pyarrow.parquet as pq  # noqa: F401
    except Exception as e:
        raise ImportError("缺少依赖 pyarrow。请先安装后重试。") from e


def inspect_parquet(
    parquet_path: Path,
    *,
    row_group: int = 0,
    max_rows: Optional[int] = None,
) -> Dict[str, Any]:
    _require_pyarrow()
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(parquet_path)
    schema_names = list(pf.schema.names)
    table = pf.read_row_group(int(row_group))
    if max_rows is not None:
        table = table.slice(0, int(max_rows))
    df = table.to_pandas()

    weight_cols = [c for c in df.columns if c.endswith("_cluster_labels_weight")]
    stats: Dict[str, Any] = {}
    for c in weight_cols:
        s = df[c]
        vals = s.dropna().to_numpy()
        if vals.size == 0:
            stats[c] = {"count": int(s.shape[0]), "non_nan": 0, "all_nan": True}
            continue
        uniq = np.unique(vals)
        stats[c] = {
            "count": int(s.shape[0]),
            "non_nan": int(vals.size),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "unique_count": int(uniq.size),
            "unique_head": [float(x) for x in uniq[:10]],
        }

    return {
        "parquet_path": str(parquet_path),
        "generated_at_utc": datetime.utcnow().isoformat() + "Z",
        "row_groups": int(pf.num_row_groups),
        "inspected_row_group": int(row_group),
        "inspected_rows": int(df.shape[0]),
        "schema_names": schema_names,
        "weight_columns": weight_cols,
        "weight_stats": stats,
    }


def parse_args():
    p = argparse.ArgumentParser(description="Inspect weight columns snapshot in a Parquet file")
    p.add_argument("--parquet_path", type=str, required=True)
    p.add_argument("--row_group", type=int, default=0)
    p.add_argument("--max_rows", type=int, default=None)
    p.add_argument("--out_json", type=str, default=None, help="可选：将摘要写入 JSON")
    p.add_argument("--log_file", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging(log_file=args.log_file)

    parquet_path = Path(args.parquet_path).expanduser().resolve()
    if not parquet_path.exists():
        raise FileNotFoundError(f"parquet 不存在: {parquet_path}")

    payload = inspect_parquet(parquet_path, row_group=args.row_group, max_rows=args.max_rows)

    print(json.dumps(payload, ensure_ascii=False, indent=2))

    if args.out_json:
        out_path = Path(args.out_json).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

