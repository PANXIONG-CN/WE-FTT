#!/usr/bin/env python3
"""
evaluation_protocol 一键入口（v1，聚合调用）。

定位：
- 这是“工作流编排器”，不承载核心算法实现（核心逻辑在各子目录脚本中）。
- 默认以“可复现/可审计”为优先：所有产物写入 evaluation_protocol/ 下的 results/figures/tables。
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


def _run(cmd: List[str]) -> None:
    # 打印可复核命令
    print("+", " ".join([f"\"{c}\"" if (" " in c or c.startswith("/")) else c for c in cmd]))
    subprocess.run(cmd, check=True)


def _weftt_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args():
    repo = _weftt_root()
    p = argparse.ArgumentParser(description="Run evaluation_protocol workflow (v1)")
    p.add_argument("--python", type=str, default=sys.executable, help="Python 可执行文件（建议使用 WE-FTT/.venv）")
    p.add_argument("--amsr2_root", type=str, required=True)
    p.add_argument("--eval_parquet", type=str, default=str(repo / "evaluation_protocol" / "datasets" / "mbt_eval_samples_v1.parquet"))
    p.add_argument("--control_dates_per_event", type=int, default=2)
    p.add_argument("--pixels_per_event_day", type=int, default=200)
    p.add_argument("--pre_days", type=int, default=20)
    p.add_argument("--post_days", type=int, default=10)
    p.add_argument("--event_ids", type=str, default=None, help="可选：评估子集 event_id（同 build_mbt_eval_dataset.py）")
    p.add_argument("--max_events", type=int, default=None)

    p.add_argument("--placebo_repeats", type=int, default=100)
    p.add_argument("--placebo_pixels_per_event_day", type=int, default=100)
    p.add_argument("--use_weights", action="store_true")

    p.add_argument("--run_era5", action="store_true")
    p.add_argument("--era5_daily_cache_dir", type=str, default=str(repo / "evaluation_protocol" / "era5" / "cache" / "daily"))
    p.add_argument("--era5_download", action="store_true", help="通过 CDS 下载ERA5并生成 daily cache（需要已配置 ~/.cdsapirc）")
    p.add_argument("--era5_cleanup_nc", action="store_true", help="生成 daily cache 后清理每月 NetCDF（节省空间）")
    p.add_argument("--era5_area", type=str, default="70,-180,-70,180", help="下载区域 bbox：N,W,S,E（默认与lat_limit一致）")

    p.add_argument("--skip_dataset", action="store_true")
    p.add_argument("--skip_placebo", action="store_true")
    p.add_argument("--skip_noise_budget", action="store_true")
    p.add_argument("--skip_land", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    # 注意：不要对 --python 做 resolve()（会跟随 venv 的 python symlink，导致跑到系统/基础解释器，
    # 进而找不到 venv site-packages）。这里仅做 expanduser + 绝对化，不解析 symlink。
    py = str(Path(args.python).expanduser().absolute())
    repo = _weftt_root()

    # 1) 事件级切分
    _run(
        [
            py,
            str(repo / "evaluation_protocol" / "data_splits" / "make_event_splits.py"),
            "--catalog_csv",
            str(repo / "data" / "raw" / "earthquake_catalog.csv"),
            "--out_path",
            str(repo / "evaluation_protocol" / "data_splits" / "event_grouped_splits_v1.json"),
            "--min_mag",
            "7.0",
            "--seed",
            "42",
            "--test_ratio",
            "0.1",
            "--val_ratio",
            "0.1",
        ]
    )

    # 2) 评估数据集（含对照窗）
    if not bool(args.skip_dataset):
        cmd = [
            py,
            str(repo / "evaluation_protocol" / "datasets" / "build_mbt_eval_dataset.py"),
            "--amsr2_root",
            str(Path(args.amsr2_root).expanduser().resolve()),
            "--out_path",
            str(Path(args.eval_parquet).expanduser()),
            "--min_mag",
            "7.0",
            "--pre_days",
            str(int(args.pre_days)),
            "--post_days",
            str(int(args.post_days)),
            "--pixels_per_event_day",
            str(int(args.pixels_per_event_day)),
            "--control_dates_per_event",
            str(int(args.control_dates_per_event)),
            "--seed",
            "42",
        ]
        if args.max_events is not None:
            cmd += ["--max_events", str(int(args.max_events))]
        if args.event_ids:
            cmd += ["--event_ids", str(args.event_ids)]
        _run(cmd)

    # 3) 噪声预算 / σ_eff
    if not bool(args.skip_noise_budget):
        _run(
            [
                py,
                str(repo / "evaluation_protocol" / "noise_budget" / "compute_sigma_eff.py"),
                "--eval_parquet",
                str(Path(args.eval_parquet).expanduser()),
            ]
        )

    # 4) Placebo（三类）
    if not bool(args.skip_placebo):
        for t in ("random_date", "time_shift", "random_location"):
            plan_path = repo / "evaluation_protocol" / "placebo" / "plans" / f"{t}_plan_v1.jsonl"
            _run(
                [
                    py,
                    str(repo / "evaluation_protocol" / "placebo" / "generate_placebos.py"),
                    "--placebo_type",
                    t,
                    "--n_repeats",
                    str(int(args.placebo_repeats)),
                    "--pre_days",
                    str(int(args.pre_days)),
                    "--post_days",
                    str(int(args.post_days)),
                    "--out_jsonl",
                    str(plan_path),
                ]
            )
            out_json = repo / "evaluation_protocol" / "placebo" / "results" / f"{t}_results_v1.json"
            cmd = [
                py,
                str(repo / "evaluation_protocol" / "placebo" / "run_placebo.py"),
                "--eval_parquet",
                str(Path(args.eval_parquet).expanduser()),
                "--amsr2_root",
                str(Path(args.amsr2_root).expanduser().resolve()),
                "--placebo_plan_jsonl",
                str(plan_path),
                "--placebo_type",
                t,
                "--n_repeats",
                str(int(args.placebo_repeats)),
                "--pre_days",
                str(int(args.pre_days)),
                "--post_days",
                str(int(args.post_days)),
                "--pixels_per_event_day",
                str(int(args.placebo_pixels_per_event_day)),
                "--control_dates_per_event",
                "1",
                "--seed",
                "42",
                "--out_json",
                str(out_json),
            ]
            if bool(args.use_weights):
                cmd += ["--use_weights"]
            _run(cmd)

    # 5) 陆地残差化
    if not bool(args.skip_land):
        _run(
            [
                py,
                str(repo / "evaluation_protocol" / "land_conditioning" / "residualize_tb.py"),
                "--eval_parquet",
                str(Path(args.eval_parquet).expanduser()),
                "--n_placebo_repeats",
                str(int(args.placebo_repeats)),
            ]
        )

    # 6) ERA5（可选）
    if bool(args.run_era5):
        if bool(args.era5_download):
            cmd = [
                py,
                str(repo / "evaluation_protocol" / "era5" / "download_from_cds.py"),
                "--eval_parquet",
                str(Path(args.eval_parquet).expanduser()),
                "--zone_type",
                "0",
                "--out_daily_dir",
                str(Path(args.era5_daily_cache_dir).expanduser()),
                "--area",
                str(args.era5_area),
            ]
            if bool(args.era5_cleanup_nc):
                cmd += ["--cleanup_nc"]
            _run(cmd)

        cmd = [
            py,
            str(repo / "evaluation_protocol" / "era5" / "ocean_conditioning.py"),
            "--eval_parquet",
            str(Path(args.eval_parquet).expanduser()),
            "--era5_daily_cache_dir",
            str(Path(args.era5_daily_cache_dir).expanduser()),
        ]
        if bool(args.use_weights):
            cmd += ["--use_weights"]
        _run(cmd)


if __name__ == "__main__":
    main()
