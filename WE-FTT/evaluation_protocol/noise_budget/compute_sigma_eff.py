#!/usr/bin/env python3
"""
T5 噪声预算/σ_eff 计算（v2，统一效应量口径）。

输出：
- Tab S8（CSV）：通道 × zone_type 的噪声估计 + 事件对照效应量 + z_eff 指标（含全局经验口径与同事件口径）

说明：
- 经验估计基于 eval_parquet 的对照窗（flag=0, window_type=control）
- σ_eff（经验-全局）使用“同一 control 窗口内所有像元×日期”的均值，再跨窗口取标准差
- σ_eff（经验-同事件）先在同一事件内对多个 control 窗口均值求标准差，再在事件间取平均
- σ_eff（经验-配对稳健）在同事件 control-control 的窗口差分绝对值上取 MAD，并按事件数换算为均值标准误
- σ_eff（理论）使用 NEDT / sqrt(mean_samples_per_window)
- 事件效应量使用同一 (zone_type,event_id) 上 event 窗均值与 control 窗均值之差
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

# 将 WE-FTT 根目录加入路径（保证可直接以脚本方式运行）
weftt_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if weftt_root not in sys.path:
    sys.path.insert(0, weftt_root)

from evaluation_protocol.common.amsr2 import FEATURE_COLUMNS  # noqa: E402
from evaluation_protocol.common.jsonl import write_json  # noqa: E402
from evaluation_protocol.common.logging_utils import setup_logging  # noqa: E402
from evaluation_protocol.common.paths import get_repo_paths, resolve_path  # noqa: E402


logger = logging.getLogger(__name__)


_NEDT_K_APPROX = {
    "06": 0.3,
    "10": 0.6,
    "23": 0.6,
    "36": 0.3,
    "89": 0.6,
}


def _nedt_for_feature(col: str) -> float:
    # col 形如 BT_06_H
    try:
        freq = col.split("_")[1]
        return float(_NEDT_K_APPROX.get(freq, np.nan))
    except Exception:
        return float("nan")


def parse_args():
    repo = get_repo_paths()
    default_eval = repo.eval_root / "datasets" / "mbt_eval_samples_v1.parquet"
    default_out = repo.eval_root / "noise_budget" / "tables" / "tab_s8_sigma_eff_v2.csv"
    default_meta = repo.eval_root / "noise_budget" / "results" / "sigma_eff_v2.meta.json"

    p = argparse.ArgumentParser(description="Compute sigma_eff table (v2)")
    p.add_argument("--eval_parquet", type=str, default=str(default_eval))
    p.add_argument("--out_csv", type=str, default=str(default_out))
    p.add_argument("--out_meta_json", type=str, default=str(default_meta))
    p.add_argument("--log_file", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging(log_file=args.log_file)

    eval_parquet = resolve_path(args.eval_parquet)
    out_csv = resolve_path(args.out_csv)
    out_meta = resolve_path(args.out_meta_json)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_meta.parent.mkdir(parents=True, exist_ok=True)

    need_cols = ["event_id", "zone_type", "flag", "window_type", "control_index", "anchor_date", *FEATURE_COLUMNS]
    df = pd.read_parquet(eval_parquet, columns=need_cols)
    df = df[(df["flag"].astype(int) == 0) & (df["window_type"].astype(str) == "control")].copy()
    if df.empty:
        raise ValueError("未找到对照窗样本（flag=0 & window_type=control）。请在构建评估集时设置 control_dates_per_event>0。")

    # 窗口均值 σ_eff：按 (zone_type,event_id,control_index,anchor_date) 聚合成窗口，再跨窗口取 std
    window_keys = ["zone_type", "event_id", "control_index", "anchor_date"]
    win = df.groupby(window_keys)[list(FEATURE_COLUMNS)].mean().reset_index()

    # 事件-对照效应量：同一 (zone_type,event_id) 下 event mean - control mean
    event_df = pd.read_parquet(eval_parquet, columns=["event_id", "zone_type", "flag", "window_type", *FEATURE_COLUMNS])
    event_df = event_df[event_df["flag"].astype(int) == 1].copy()
    if event_df.empty:
        raise ValueError("未找到事件窗样本（flag=1）。")
    ev_mean = event_df.groupby(["zone_type", "event_id"])[list(FEATURE_COLUMNS)].mean()
    ctrl_mean = df.groupby(["zone_type", "event_id"])[list(FEATURE_COLUMNS)].mean()
    common_idx = ev_mean.index.intersection(ctrl_mean.index)
    delta = (ev_mean.loc[common_idx] - ctrl_mean.loc[common_idx]).copy()

    # 组装长表
    rows: List[Dict] = []
    zone_types = sorted(df["zone_type"].dropna().unique().tolist())
    zones_with_delta = set(delta.index.get_level_values(0).tolist())
    for z in zone_types:
        df_z = df[df["zone_type"] == z]
        win_z = win[win["zone_type"] == z]
        n_samples = int(len(df_z))
        n_windows = int(len(win_z))
        # 每个窗口样本量（经验）
        win_counts = df_z.groupby(window_keys).size().to_numpy()
        mean_n_per_window = float(np.mean(win_counts)) if win_counts.size else float("nan")
        delta_z = delta.loc[z] if z in zones_with_delta else None
        n_events_with_pairs = int(len(delta_z)) if delta_z is not None else 0
        # 经验 σ_eff（同事件口径）：每个事件内对 control 窗口均值求 std，再跨事件取均值
        # 若事件仅有 1 个 control 窗口，该事件对应 std 为 NaN，不计入均值
        event_ctrl_std = win_z.groupby("event_id")[list(FEATURE_COLUMNS)].std(ddof=1)
        # 经验 σ_eff（配对稳健口径）：同事件 control-control 窗口差分绝对值（null）
        # 再使用 MAD 作为稳健尺度估计，按事件数换算为均值标准误
        null_abs_pairs: Dict[str, List[float]] = {col: [] for col in FEATURE_COLUMNS}
        for _, eg in win_z.groupby("event_id"):
            arr = eg[list(FEATURE_COLUMNS)].to_numpy(dtype="float64")
            if arr.shape[0] < 2:
                continue
            for i in range(arr.shape[0] - 1):
                for j in range(i + 1, arr.shape[0]):
                    diff_abs = np.abs(arr[i] - arr[j])
                    for k, col in enumerate(FEATURE_COLUMNS):
                        val = float(diff_abs[k])
                        if np.isfinite(val):
                            null_abs_pairs[col].append(val)

        for col in FEATURE_COLUMNS:
            s = df_z[col].astype("float64").to_numpy()
            sigma_sample = float(np.nanstd(s, ddof=1)) if np.sum(~np.isnan(s)) >= 2 else float("nan")

            w = win_z[col].astype("float64").to_numpy()
            sigma_win_mean = float(np.nanstd(w, ddof=1)) if np.sum(~np.isnan(w)) >= 2 else float("nan")
            event_std_col = event_ctrl_std[col].astype("float64").to_numpy()
            sigma_within_event = float(np.nanmean(event_std_col)) if np.sum(np.isfinite(event_std_col)) >= 1 else float("nan")
            n_events_within_event_sigma = int(np.sum(np.isfinite(event_std_col)))
            nedt = float(_nedt_for_feature(col))
            sigma_theory = float(nedt / np.sqrt(mean_n_per_window)) if np.isfinite(mean_n_per_window) and mean_n_per_window > 0 else float("nan")
            null_vals = np.asarray(null_abs_pairs.get(col, []), dtype=np.float64)
            null_vals = null_vals[np.isfinite(null_vals)]
            null_median = float(np.median(null_vals)) if null_vals.size else float("nan")
            null_mad = (
                float(np.median(np.abs(null_vals - null_median)) * 1.4826)
                if null_vals.size
                else float("nan")
            )

            if delta_z is not None and col in delta_z.columns:
                d = delta_z[col].astype("float64").to_numpy()
                d = d[np.isfinite(d)]
            else:
                d = np.asarray([], dtype=np.float64)
            mean_delta = float(np.mean(d)) if d.size else float("nan")
            mean_abs_delta = float(np.mean(np.abs(d))) if d.size else float("nan")
            sigma_paired_robust = (
                float(null_mad / np.sqrt(d.size))
                if d.size and np.isfinite(null_mad) and null_mad > 0
                else float("nan")
            )
            z_abs_emp = float(mean_abs_delta / sigma_win_mean) if np.isfinite(mean_abs_delta) and np.isfinite(sigma_win_mean) and sigma_win_mean > 0 else float("nan")
            z_abs_emp_within = (
                float(mean_abs_delta / sigma_within_event)
                if np.isfinite(mean_abs_delta) and np.isfinite(sigma_within_event) and sigma_within_event > 0
                else float("nan")
            )
            z_abs_emp_paired_robust = (
                float(mean_abs_delta / sigma_paired_robust)
                if np.isfinite(mean_abs_delta) and np.isfinite(sigma_paired_robust) and sigma_paired_robust > 0
                else float("nan")
            )
            z_abs_theory = float(mean_abs_delta / sigma_theory) if np.isfinite(mean_abs_delta) and np.isfinite(sigma_theory) and sigma_theory > 0 else float("nan")
            z_mean_emp = float(abs(mean_delta) / sigma_win_mean) if np.isfinite(mean_delta) and np.isfinite(sigma_win_mean) and sigma_win_mean > 0 else float("nan")
            frac_gt2_emp = (
                float(np.mean(np.abs(d) > 2.0 * sigma_win_mean))
                if d.size and np.isfinite(sigma_win_mean) and sigma_win_mean > 0
                else float("nan")
            )
            frac_gt2_emp_within = (
                float(np.mean(np.abs(d) > 2.0 * sigma_within_event))
                if d.size and np.isfinite(sigma_within_event) and sigma_within_event > 0
                else float("nan")
            )
            frac_gt2_emp_paired_robust = (
                float(np.mean(np.abs(d) > 2.0 * sigma_paired_robust))
                if d.size and np.isfinite(sigma_paired_robust) and sigma_paired_robust > 0
                else float("nan")
            )
            frac_gt2_theory = (
                float(np.mean(np.abs(d) > 2.0 * sigma_theory))
                if d.size and np.isfinite(sigma_theory) and sigma_theory > 0
                else float("nan")
            )

            rows.append(
                {
                    "zone_type": int(z),
                    "feature": str(col),
                    "nedt_k_approx": float(nedt),
                    "n_control_samples": int(n_samples),
                    "n_control_windows": int(n_windows),
                    "mean_samples_per_window": float(mean_n_per_window),
                    "sigma_control_sample_k": float(sigma_sample),
                    "sigma_control_window_mean_k": float(sigma_win_mean),
                    "sigma_control_within_event_k": float(sigma_within_event),
                    "n_events_within_event_sigma": int(n_events_within_event_sigma),
                    "n_control_null_pairs_within_event": int(null_vals.size),
                    "sigma_control_paired_robust_mean_se_k": float(sigma_paired_robust),
                    "sigma_eff_theory_k": float(sigma_theory),
                    "n_events_with_pairs": int(n_events_with_pairs),
                    "mean_event_control_delta_k": float(mean_delta),
                    "mean_abs_event_control_delta_k": float(mean_abs_delta),
                    "z_eff_abs_vs_empirical_sigma": float(z_abs_emp),
                    "z_eff_abs_vs_empirical_within_event_sigma": float(z_abs_emp_within),
                    "z_eff_abs_vs_empirical_paired_robust_sigma": float(z_abs_emp_paired_robust),
                    "z_eff_abs_vs_theory_sigma": float(z_abs_theory),
                    "z_eff_mean_vs_empirical_sigma": float(z_mean_emp),
                    "frac_events_abs_delta_gt_2sigma_empirical": float(frac_gt2_emp),
                    "frac_events_abs_delta_gt_2sigma_empirical_within_event": float(frac_gt2_emp_within),
                    "frac_events_abs_delta_gt_2sigma_empirical_paired_robust": float(frac_gt2_emp_paired_robust),
                    "frac_events_abs_delta_gt_2sigma_theory": float(frac_gt2_theory),
                    "criterion_gt_2sigma_empirical": bool(np.isfinite(z_abs_emp) and z_abs_emp >= 2.0),
                    "criterion_gt_2sigma_empirical_within_event": bool(np.isfinite(z_abs_emp_within) and z_abs_emp_within >= 2.0),
                    "criterion_gt_2sigma_empirical_paired_robust": bool(
                        np.isfinite(z_abs_emp_paired_robust) and z_abs_emp_paired_robust >= 2.0
                    ),
                    "criterion_gt_2sigma_theory": bool(np.isfinite(z_abs_theory) and z_abs_theory >= 2.0),
                }
            )

    out_df = pd.DataFrame(rows).sort_values(["zone_type", "feature"])
    out_df.to_csv(out_csv, index=False)

    write_json(
        out_meta,
        {
            "version": "sigma_eff_v2",
            "generated_at_utc": datetime.utcnow().isoformat() + "Z",
            "eval_parquet": str(eval_parquet),
            "out_csv": str(out_csv),
            "notes": {
                "nedt_k_approx": "来自任务计划文档中的近似值（用于理论量级对照）",
                "sigma_control_sample_k": "对照窗样本级标准差（经验）",
                "sigma_control_window_mean_k": "对照窗窗口均值的标准差（经验 σ_eff，全局口径）",
                "sigma_control_within_event_k": "同事件内 control 窗口均值标准差的事件均值（经验 σ_eff，同事件口径）",
                "sigma_control_paired_robust_mean_se_k": "同事件 control-control 差分绝对值的 MAD，并按事件数换算的稳健均值标准误",
                "sigma_eff_theory_k": "NEDT/sqrt(mean_samples_per_window) 的理论噪声下界",
                "mean_abs_event_control_delta_k": "同区同事件 event/control 窗口均值差的绝对值均值",
                "z_eff_abs_vs_empirical_sigma": "mean_abs_event_control_delta_k / sigma_control_window_mean_k",
                "z_eff_abs_vs_empirical_within_event_sigma": "mean_abs_event_control_delta_k / sigma_control_within_event_k",
                "z_eff_abs_vs_empirical_paired_robust_sigma": "mean_abs_event_control_delta_k / sigma_control_paired_robust_mean_se_k",
                "z_eff_abs_vs_theory_sigma": "mean_abs_event_control_delta_k / sigma_eff_theory_k",
            },
        },
    )
    logger.info("写入: %s", out_csv)


if __name__ == "__main__":
    main()
