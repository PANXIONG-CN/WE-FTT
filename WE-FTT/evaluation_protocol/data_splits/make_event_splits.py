#!/usr/bin/env python3
"""
生成事件级（event-grouped）切分清单。

输出：event_grouped_splits_v1.json
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# 将 WE-FTT 根目录加入路径
weftt_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if weftt_root not in sys.path:
    sys.path.insert(0, weftt_root)

from evaluation_protocol.common.catalog import load_events  # noqa: E402
from evaluation_protocol.common.jsonl import write_json  # noqa: E402
from evaluation_protocol.common.logging_utils import setup_logging  # noqa: E402
from evaluation_protocol.common.paths import get_repo_paths, resolve_path  # noqa: E402


logger = logging.getLogger(__name__)


def split_by_zone(event_ids_by_zone: Dict[int, List[str]], *, seed: int, test_ratio: float, val_ratio: float):
    rng = np.random.default_rng(int(seed))
    train, val, test = [], [], []
    train_by_zone: Dict[int, List[str]] = {}
    val_by_zone: Dict[int, List[str]] = {}
    test_by_zone: Dict[int, List[str]] = {}

    for z, ids in sorted(event_ids_by_zone.items(), key=lambda x: x[0]):
        ids = list(ids)
        rng.shuffle(ids)
        n = len(ids)
        if n < 3:
            # 太小则全部进 train（保持定义简单）
            train_by_zone[z] = ids
            val_by_zone[z] = []
            test_by_zone[z] = []
            train.extend(ids)
            continue

        n_test = int(round(n * float(test_ratio)))
        n_val = int(round(n * float(val_ratio)))
        # 尽量保证 val/test 至少各 1（在 n>=4 时）
        n_test = max(1, n_test)
        n_val = max(1, n_val) if n >= 4 else max(0, n_val)

        # 防止挤占 train
        if n_test + n_val >= n:
            # 最小保留 1 个 train
            if n >= 5:
                n_test = 1
                n_val = 1
            else:
                n_test = 1
                n_val = 0

        test_ids = ids[:n_test]
        val_ids = ids[n_test : n_test + n_val]
        train_ids = ids[n_test + n_val :]

        test_by_zone[z] = test_ids
        val_by_zone[z] = val_ids
        train_by_zone[z] = train_ids

        test.extend(test_ids)
        val.extend(val_ids)
        train.extend(train_ids)

    return {
        "train_event_ids": train,
        "val_event_ids": val,
        "test_event_ids": test,
        "by_zone": {
            "train": train_by_zone,
            "val": val_by_zone,
            "test": test_by_zone,
        },
    }


def parse_args():
    repo = get_repo_paths()
    default_catalog = repo.weftt_root / "data" / "raw" / "earthquake_catalog.csv"
    default_out = repo.eval_root / "data_splits" / "event_grouped_splits_v1.json"

    p = argparse.ArgumentParser(description="Make event-grouped splits (v1)")
    p.add_argument("--catalog_csv", type=str, default=str(default_catalog))
    p.add_argument("--out_path", type=str, default=str(default_out))
    p.add_argument("--min_mag", type=float, default=7.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test_ratio", type=float, default=0.1)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--log_file", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging(log_file=args.log_file)

    catalog_csv = resolve_path(args.catalog_csv)
    out_path = resolve_path(args.out_path)

    events = load_events(catalog_csv, min_mag=args.min_mag, require_zone=True)
    logger.info("事件数（M>=%.1f）：%d", args.min_mag, len(events))

    by_zone: Dict[int, List[str]] = defaultdict(list)
    for ev in events:
        if ev.zone_type is None:
            continue
        by_zone[int(ev.zone_type)].append(ev.event_id)

    split = split_by_zone(by_zone, seed=args.seed, test_ratio=args.test_ratio, val_ratio=args.val_ratio)

    payload = {
        "version": "event_grouped_splits_v1",
        "generated_at_utc": datetime.utcnow().isoformat() + "Z",
        "seed": int(args.seed),
        "criteria": {
            "catalog_csv": str(catalog_csv),
            "min_mag": float(args.min_mag),
            "marine_depth_max_km": 70.0,
            "marine_zone_type": 0,
            "require_zone": True,
        },
        "ratios": {"test": float(args.test_ratio), "val": float(args.val_ratio), "train": 1.0 - float(args.test_ratio) - float(args.val_ratio)},
        "train_event_ids": split["train_event_ids"],
        "val_event_ids": split["val_event_ids"],
        "test_event_ids": split["test_event_ids"],
        "by_zone": split["by_zone"],
    }

    write_json(out_path, payload)
    logger.info("写入: %s", out_path)
    logger.info("split sizes: train=%d val=%d test=%d", len(payload["train_event_ids"]), len(payload["val_event_ids"]), len(payload["test_event_ids"]))


if __name__ == "__main__":
    main()

