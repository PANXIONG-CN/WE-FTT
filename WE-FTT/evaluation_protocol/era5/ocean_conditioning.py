#!/usr/bin/env python3
"""
T6 海洋区（Zone A / type=0）ERA5 条件化：硬掩膜（v1）。

输入：
- eval_parquet：带 grid_i/grid_j/sample_date/event_id/flag 的评估数据
- ERA5 日缓存：wind_speed_YYYYMMDD.npy、precip_mm_YYYYMMDD.npy（由 download_and_align.py 生成）

输出：
- Tab S7：掩膜前后 FPR（含 Wilson CI）与样本量对比
- Fig S6：掩膜前后 FPR 对比柱状图
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# 将 WE-FTT 根目录加入路径（保证可直接以脚本方式运行）
weftt_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if weftt_root not in sys.path:
    sys.path.insert(0, weftt_root)

from evaluation_protocol.common.amsr2 import FEATURE_COLUMNS  # noqa: E402
from evaluation_protocol.common.jsonl import write_json  # noqa: E402
from evaluation_protocol.common.logging_utils import setup_logging  # noqa: E402
from evaluation_protocol.common.metrics import confusion_from_binary, fpr, matthews_corrcoef, wilson_ci  # noqa: E402
from evaluation_protocol.common.paths import get_repo_paths, resolve_path  # noqa: E402
from evaluation_protocol.common.splits import load_event_splits, split_df_by_event_id  # noqa: E402
from evaluation_protocol.common.weighting import add_foldwise_kmeans_weights, apply_weighting_artifacts  # noqa: E402


logger = logging.getLogger(__name__)


def _prepare_xy(df: pd.DataFrame, *, use_weights: bool) -> tuple[np.ndarray, np.ndarray]:
    y = df["flag"].astype("int64").to_numpy()
    x_feat = df[list(FEATURE_COLUMNS)].astype("float32").to_numpy()
    if not use_weights:
        return x_feat, y
    w_cols = [f"{c}_cluster_labels_weight" for c in FEATURE_COLUMNS]
    x_w = df[w_cols].astype("float32").to_numpy()
    return (x_feat * x_w), y


def train_classifier(train_df: pd.DataFrame, *, use_weights: bool, seed: int) -> Pipeline:
    x, y = _prepare_xy(train_df, use_weights=use_weights)
    clf = SGDClassifier(
        loss="log_loss",
        alpha=1e-4,
        max_iter=2000,
        tol=1e-3,
        class_weight="balanced",
        random_state=int(seed),
    )
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    pipe.fit(x, y)
    return pipe


def eval_metrics(pipe: Pipeline, df: pd.DataFrame, *, use_weights: bool) -> Dict[str, float]:
    x, y_true = _prepare_xy(df, use_weights=use_weights)
    proba = pipe.predict_proba(x)[:, 1]
    y_pred = (proba >= 0.5).astype(np.int64)
    c = confusion_from_binary(y_true, y_pred)
    lo, hi = wilson_ci(successes=c.fp, n=(c.fp + c.tn))
    return {
        "mcc": float(matthews_corrcoef(c)),
        "fpr": float(fpr(c)),
        "fpr_ci_low": float(lo),
        "fpr_ci_high": float(hi),
        "tp": int(c.tp),
        "fp": int(c.fp),
        "tn": int(c.tn),
        "fn": int(c.fn),
        "n": int(len(df)),
    }


def _load_daily_cache(cache_dir: Path, prefix: str, ymd: str) -> np.ndarray:
    fp = cache_dir / f"{prefix}_{ymd}.npy"
    if not fp.exists():
        raise FileNotFoundError(fp)
    return np.load(fp)


def apply_era5_mask(
    df: pd.DataFrame,
    *,
    cache_dir: Path,
    wind_thresh: float,
    precip_thresh: float,
) -> pd.DataFrame:
    """
    对每行样本按 (sample_date, grid_i, grid_j) 取 ERA5 值并做硬掩膜。
    """
    if df.empty:
        return df
    df = df.reset_index(drop=True)
    # 按日期批处理，避免重复加载
    keep = np.ones(len(df), dtype=bool)
    for d, idx in df.groupby("sample_date").groups.items():
        ymd = str(d).replace("-", "")
        try:
            ws = _load_daily_cache(cache_dir, "wind_speed", ymd)
            pr = _load_daily_cache(cache_dir, "precip_mm", ymd)
        except FileNotFoundError as e:
            # 缓存缺失时不做掩膜（保守：不引入额外选择偏差）
            logger.warning("ERA5 缓存缺失（跳过掩膜，保留该日样本）：%s", e)
            continue
        ii = df.loc[idx, "grid_i"].astype(int).to_numpy()
        jj = df.loc[idx, "grid_j"].astype(int).to_numpy()
        ws_v = ws[ii, jj]
        pr_v = pr[ii, jj]
        ok = ((ws_v <= float(wind_thresh)) | np.isnan(ws_v)) & (
            (pr_v <= float(precip_thresh)) | np.isnan(pr_v)
        )
        keep[idx] = ok
    return df[keep].copy()


def parse_args():
    repo = get_repo_paths()
    default_eval = repo.eval_root / "datasets" / "mbt_eval_samples_v1.parquet"
    default_splits = repo.eval_root / "data_splits" / "event_grouped_splits_v1.json"
    default_cache = repo.eval_root / "era5" / "cache" / "daily"
    default_out = repo.eval_root / "era5" / "results" / "ocean_conditioning_v1.json"
    default_tab = repo.eval_root / "era5" / "tables" / "tab_s7_era5_ocean_mask_v1.csv"
    default_fig = repo.eval_root / "era5" / "figures" / "fig_s6_era5_ocean_fpr_v1.png"

    p = argparse.ArgumentParser(description="ERA5 ocean conditioning (hard mask, v1)")
    p.add_argument("--eval_parquet", type=str, default=str(default_eval))
    p.add_argument("--splits_json", type=str, default=str(default_splits))
    p.add_argument("--era5_daily_cache_dir", type=str, default=str(default_cache))
    p.add_argument("--wind_thresh", type=float, default=10.0, help="m/s")
    p.add_argument("--precip_thresh", type=float, default=5.0, help="mm/day")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use_weights", action="store_true")
    p.add_argument("--n_clusters", type=int, default=5)
    p.add_argument("--out_json", type=str, default=str(default_out))
    p.add_argument("--out_tab_csv", type=str, default=str(default_tab))
    p.add_argument("--out_fig_png", type=str, default=str(default_fig))
    p.add_argument("--log_file", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging(log_file=args.log_file)

    eval_parquet = resolve_path(args.eval_parquet)
    splits_json = resolve_path(args.splits_json)
    cache_dir = resolve_path(args.era5_daily_cache_dir)
    out_json = resolve_path(args.out_json)
    out_tab = resolve_path(args.out_tab_csv)
    out_fig = resolve_path(args.out_fig_png)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_tab.parent.mkdir(parents=True, exist_ok=True)
    out_fig.parent.mkdir(parents=True, exist_ok=True)

    need_cols = ["event_id", "zone_type", "flag", "sample_date", "grid_i", "grid_j", *FEATURE_COLUMNS]
    df = pd.read_parquet(eval_parquet, columns=need_cols)
    df = df[df["zone_type"].astype(int) == 0].copy()
    if df.empty:
        raise ValueError("eval_parquet 中未找到 Zone A（zone_type=0）样本。")

    splits = load_event_splits(splits_json)
    train_df, val_df, test_df = split_df_by_event_id(df, splits)

    artifacts = []
    if bool(args.use_weights):
        train_df, (val_df, test_df), artifacts = add_foldwise_kmeans_weights(
            train_df,
            [val_df, test_df],
            feature_columns=FEATURE_COLUMNS,
            n_clusters=int(args.n_clusters),
            seed=int(args.seed),
        )

    pipe = train_classifier(train_df, use_weights=bool(args.use_weights), seed=int(args.seed))

    before = eval_metrics(pipe, test_df, use_weights=bool(args.use_weights))
    masked_test = apply_era5_mask(
        test_df,
        cache_dir=cache_dir,
        wind_thresh=float(args.wind_thresh),
        precip_thresh=float(args.precip_thresh),
    )
    if bool(args.use_weights) and artifacts:
        masked_test = apply_weighting_artifacts(masked_test, artifacts)
    after = eval_metrics(pipe, masked_test, use_weights=bool(args.use_weights))

    pd.DataFrame(
        [
            {
                "subset": "zone_a_test",
                "wind_thresh_mps": float(args.wind_thresh),
                "precip_thresh_mm_day": float(args.precip_thresh),
                "n_before": int(before["n"]),
                "n_after": int(after["n"]),
                "fpr_before": float(before["fpr"]),
                "fpr_before_ci_low": float(before["fpr_ci_low"]),
                "fpr_before_ci_high": float(before["fpr_ci_high"]),
                "fpr_after": float(after["fpr"]),
                "fpr_after_ci_low": float(after["fpr_ci_low"]),
                "fpr_after_ci_high": float(after["fpr_ci_high"]),
                "mcc_before": float(before["mcc"]),
                "mcc_after": float(after["mcc"]),
            }
        ]
    ).to_csv(out_tab, index=False)

    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(5.5, 3.2))
        xs = ["Before", "After"]
        ys = [before["fpr"], after["fpr"]]
        yerr = [
            [before["fpr"] - before["fpr_ci_low"], after["fpr"] - after["fpr_ci_low"]],
            [before["fpr_ci_high"] - before["fpr"], after["fpr_ci_high"] - after["fpr"]],
        ]
        plt.bar(xs, ys, yerr=yerr, capsize=5)
        plt.ylim(0, 1)
        plt.ylabel("FPR (Zone A)")
        plt.title("ERA5 Hard Mask Conditioning")
        plt.tight_layout()
        plt.savefig(out_fig, dpi=200)
        plt.close()
    except Exception as e:
        logger.warning("绘图失败：%s", e)

    payload = {
        "version": "ocean_conditioning_v1",
        "generated_at_utc": datetime.utcnow().isoformat() + "Z",
        "inputs": {"eval_parquet": str(eval_parquet), "splits_json": str(splits_json), "era5_daily_cache_dir": str(cache_dir)},
        "params": {"wind_thresh": float(args.wind_thresh), "precip_thresh": float(args.precip_thresh), "seed": int(args.seed), "use_weights": bool(args.use_weights)},
        "before": before,
        "after": after,
        "artifacts": {"table_csv": str(out_tab), "figure_png": str(out_fig)},
    }
    write_json(out_json, payload)
    logger.info("写入: %s", out_json)


if __name__ == "__main__":
    main()
