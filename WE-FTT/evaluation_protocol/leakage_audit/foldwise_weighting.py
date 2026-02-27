#!/usr/bin/env python3
"""
按事件级切分（event-grouped split）进行“训练折内拟合→应用到val/test”的权重生成（v1）。

目的：
- 满足 T1 的 DoD：先切分 → 仅训练折内拟合 → 权重应用到 val/test
- 将“权重生成”的可疑泄漏风险，变成可审计、可复现的产物（模型/映射落盘）

实现选择（KISS）：
- 每个通道单独做 KMeans 离散化（仅拟合 train）
- 每个簇计算训练集上的 support_diff：|P(cluster|pos)-P(cluster|neg)|
- 按簇ID为样本写入 `{feature}_cluster_labels_weight` 列（0~1 归一化）

注意：
- 这是“最小可用”的泄漏修复实现，用于评估协议；不尝试复刻复杂的多项集关联规则。
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
import pandas as pd

# 将 WE-FTT 根目录加入路径（保证可直接以脚本方式运行）
weftt_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if weftt_root not in sys.path:
    sys.path.insert(0, weftt_root)

from evaluation_protocol.common.amsr2 import FEATURE_COLUMNS  # noqa: E402
from evaluation_protocol.common.jsonl import write_json  # noqa: E402
from evaluation_protocol.common.logging_utils import setup_logging  # noqa: E402
from evaluation_protocol.common.paths import get_repo_paths, resolve_path  # noqa: E402
from evaluation_protocol.common.splits import load_event_splits, split_df_by_event_id  # noqa: E402
from evaluation_protocol.common.weighting import (  # noqa: E402
    add_foldwise_kmeans_weights,
    artifacts_to_jsonable,
)


def parse_args():
    repo = get_repo_paths()
    p = argparse.ArgumentParser(description="Leak-free fold-wise weighting (event-grouped)")
    p.add_argument("--in_parquet", type=str, required=True, help="输入评估数据集（需含 event_id/flag/10通道）")
    p.add_argument("--splits_json", type=str, default=str(repo.eval_root / "data_splits" / "event_grouped_splits_v1.json"))
    p.add_argument("--out_dir", type=str, default=str(repo.eval_root / "leakage_audit" / "traces" / "foldwise_weights_v1"))
    p.add_argument("--n_clusters", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=65536)
    p.add_argument("--log_file", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging(log_file=args.log_file)

    in_parquet = resolve_path(args.in_parquet)
    splits_json = resolve_path(args.splits_json)
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = load_event_splits(splits_json)

    df = pd.read_parquet(in_parquet, columns=["event_id", "flag", "label", "zone_type", *FEATURE_COLUMNS])
    train_df, val_df, test_df = split_df_by_event_id(df, splits)
    train_df, (val_df, test_df), artifacts = add_foldwise_kmeans_weights(
        train_df,
        [val_df, test_df],
        feature_columns=FEATURE_COLUMNS,
        n_clusters=int(args.n_clusters),
        seed=int(args.seed),
        batch_size=int(args.batch_size),
    )

    train_path = out_dir / "train_with_weights.parquet"
    val_path = out_dir / "val_with_weights.parquet"
    test_path = out_dir / "test_with_weights.parquet"

    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    test_df.to_parquet(test_path, index=False)

    meta_path = out_dir / "artifacts.meta.json"
    write_json(
        meta_path,
        {
            "version": "foldwise_weights_v1",
            "generated_at_utc": datetime.utcnow().isoformat() + "Z",
            "in_parquet": str(in_parquet),
            "splits_json": str(splits_json),
            "n_clusters": int(args.n_clusters),
            "seed": int(args.seed),
            "batch_size": int(args.batch_size),
            "outputs": {
                "train": str(train_path),
                "val": str(val_path),
                "test": str(test_path),
            },
            "artifacts": artifacts_to_jsonable(artifacts),
            "sizes": {"train": int(len(train_df)), "val": int(len(val_df)), "test": int(len(test_df))},
        },
    )


if __name__ == "__main__":
    main()
