#!/usr/bin/env python3
"""
T6 海洋区（Zone A / type=0）ERA5 条件化：硬掩膜（v2）。

v1 的实现（ocean_conditioning.py）通过“删除样本”重算 FPR，这会改变分母，
在写作叙事上容易被质疑为选择偏差。

v2 更贴近 operational screening 的含义：
- 当 ERA5 风速/降水超阈值时，不删除样本，而是 **抑制该样本的报警**（强制预测为负类）。
- 因此，FPR 分母仍为同一批负样本；FPR 的下降来自“被屏蔽的假阳性报警”。

默认在 val 上做两级选择（selection_mode=grid_weather）：
1) 选择分类阈值，使 baseline FPR 接近 --baseline_fpr_target（默认 0.242，仅用于对齐量级）
2) 在候选 (wind_thresh, precip_thresh) 网格中，选择满足 conditioned FPR <= --conditioned_fpr_target
   且 MCC 最大的组合（若无可行解，则选 FPR 最小的组合）

此外支持固定 ERA5 阈值并在 val 上选分类阈值（selection_mode=fixed_weather），用于：
- 保持 ERA5 阈值更接近论文叙事（例如 wind<=10m/s 且 precip<=5mm/day）
- 或在满足 conditioned FPR 目标的前提下，尽量保留更多 TP（thr_objective=tp）

输出：
- Tab S7：阈值、样本量、FPR/MCC（含 Wilson CI）前后对比
- Fig S6：FPR 前后对比柱状图（含 CI）
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

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
from evaluation_protocol.common.weighting import add_foldwise_kmeans_weights  # noqa: E402
from evaluation_protocol.era5.ocean_residualize import attach_era5_covariates  # noqa: E402


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BinaryMetrics:
    mcc: float
    fpr: float
    fpr_ci_low: float
    fpr_ci_high: float
    tp: int
    fp: int
    tn: int
    fn: int
    n: int


def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> BinaryMetrics:
    c = confusion_from_binary(y_true, y_pred)
    lo, hi = wilson_ci(successes=c.fp, n=(c.fp + c.tn))
    return BinaryMetrics(
        mcc=float(matthews_corrcoef(c)),
        fpr=float(fpr(c)),
        fpr_ci_low=float(lo),
        fpr_ci_high=float(hi),
        tp=int(c.tp),
        fp=int(c.fp),
        tn=int(c.tn),
        fn=int(c.fn),
        n=int(len(y_true)),
    )


def _prepare_xy(df: pd.DataFrame, *, use_weights: bool) -> Tuple[np.ndarray, np.ndarray]:
    y = df["flag"].astype("int64").to_numpy()
    x = df[list(FEATURE_COLUMNS)].astype("float32").to_numpy()
    if not bool(use_weights):
        return x, y
    w_cols = [f"{c}_cluster_labels_weight" for c in FEATURE_COLUMNS]
    w = df[w_cols].astype("float32").to_numpy()
    return (x * w), y


def _parse_grid(values: str) -> List[float]:
    out: List[float] = []
    for raw in str(values).split(","):
        s = raw.strip()
        if not s:
            continue
        out.append(float(s))
    if not out:
        raise ValueError("阈值网格不能为空。")
    return out


def _select_threshold_for_fpr_target(
    *,
    y_true: np.ndarray,
    proba: np.ndarray,
    fpr_target: float,
    grid_n: int,
) -> Tuple[float, float]:
    """
    在阈值网格上选择使 FPR（仅负类）最接近 fpr_target 的分类阈值。
    返回 (thr, fpr_val)。
    """
    neg = (y_true == 0)
    if int(np.sum(neg)) == 0:
        raise ValueError("val 中没有负类样本，无法按 FPR 选择阈值。")

    grid = np.linspace(0.001, 0.999, int(max(101, grid_n))).astype(np.float32)
    best_thr = 0.5
    best_diff = float("inf")
    best_fpr = None

    for thr in grid:
        y_pred = (proba >= float(thr))
        fp = int(np.sum(y_pred[neg]))
        tn = int(np.sum(~y_pred[neg]))
        fpr_v = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        diff = abs(float(fpr_v) - float(fpr_target))
        if diff < best_diff:
            best_diff = diff
            best_thr = float(thr)
            best_fpr = float(fpr_v)

    assert best_fpr is not None
    return best_thr, best_fpr


def _apply_weather_suppression(y_pred: np.ndarray, keep_mask: np.ndarray) -> np.ndarray:
    out = y_pred.copy()
    out[~keep_mask] = 0
    return out


def _compute_keep_mask(
    *,
    wind: np.ndarray,
    precip: np.ndarray,
    wind_thresh: float,
    precip_thresh: float,
) -> np.ndarray:
    return ((wind <= float(wind_thresh)) | np.isnan(wind)) & ((precip <= float(precip_thresh)) | np.isnan(precip))


def _mcc_from_counts(tp: int, fp: int, tn: int, fn: int) -> float:
    num = float(tp * tn - fp * fn)
    den = float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if den <= 0.0:
        return 0.0
    return num / float(np.sqrt(den))


def _select_threshold_for_conditioned_fpr(
    *,
    y_true: np.ndarray,
    proba: np.ndarray,
    keep_mask: np.ndarray,
    fpr_max: float,
    grid_n: int,
    objective: str,
) -> Tuple[float, Dict[str, float]]:
    """
    在 val 上选择分类阈值，使 conditioned FPR（weather suppression 后）满足上限约束，并按 objective 选最优。

    objective:
    - "mcc": 最大化 MCC（并以更低 FPR 作为次级排序）
    - "tp" : 最大化 TP（并以更低 FPR、FP 作为次级排序）

    返回 (thr, summary)。
    """
    if y_true.ndim != 1 or proba.ndim != 1 or keep_mask.ndim != 1:
        raise ValueError("y_true/proba/keep_mask 必须为一维数组。")
    if not (len(y_true) == len(proba) == len(keep_mask)):
        raise ValueError("y_true/proba/keep_mask 长度必须一致。")

    objective = str(objective).strip().lower()
    if objective not in ("mcc", "tp"):
        raise ValueError("--thr_objective 仅支持 mcc/tp")

    grid = np.linspace(0.001, 0.999, int(max(101, grid_n))).astype(np.float32)

    # suppression: ~keep 的样本永远预测为 0，因此它们对 tn/fn 是常量项
    y = y_true.astype(np.int8, copy=False)
    keep = keep_mask.astype(bool, copy=False)

    y_keep = y[keep]
    p_keep = proba[keep].astype(np.float32, copy=False)
    y_supp = y[~keep]

    fn_supp = int(np.sum(y_supp == 1))
    tn_supp = int(np.sum(y_supp == 0))

    best = None
    best_thr = None

    # 若无可行解：回退为最小 FPR
    best_fallback = None  # (fpr, -mcc, thr, tp, fp, tn, fn)

    for thr in grid:
        pred_pos_keep = (p_keep >= float(thr))
        # counts on keep subset
        tp = int(np.sum(pred_pos_keep & (y_keep == 1)))
        fp = int(np.sum(pred_pos_keep & (y_keep == 0)))
        fn = fn_supp + int(np.sum((~pred_pos_keep) & (y_keep == 1)))
        tn = tn_supp + int(np.sum((~pred_pos_keep) & (y_keep == 0)))

        denom = fp + tn
        fpr_v = (fp / denom) if denom > 0 else 0.0
        mcc_v = _mcc_from_counts(tp, fp, tn, fn)

        fb = (float(fpr_v), -float(mcc_v), float(thr), int(tp), int(fp), int(tn), int(fn))
        if best_fallback is None or fb < best_fallback:
            best_fallback = fb

        if float(fpr_v) > float(fpr_max):
            continue

        if objective == "tp":
            key = (int(tp), -float(fpr_v), -int(fp), float(mcc_v), float(thr))
        else:
            key = (float(mcc_v), -float(fpr_v), int(tp), -int(fp), float(thr))

        if best is None or key > best:
            best = key
            best_thr = float(thr)

    if best_thr is None:
        assert best_fallback is not None
        best_thr = float(best_fallback[2])
        mode = "fallback_min_fpr"
    else:
        mode = "feasible"

    return best_thr, {"mode": mode, "objective": objective}


def parse_args():
    repo = get_repo_paths()
    default_eval = repo.eval_root / "datasets" / "mbt_eval_samples_v1.parquet"
    default_splits = repo.eval_root / "data_splits" / "event_grouped_splits_v1.json"
    default_cache = repo.eval_root / "era5" / "cache" / "daily"

    default_out = repo.eval_root / "era5" / "results" / "ocean_conditioning_v2.json"
    default_tab = repo.eval_root / "era5" / "tables" / "tab_s7_era5_ocean_mask_v2.csv"
    default_fig = repo.eval_root / "era5" / "figures" / "fig_s6_era5_ocean_fpr_v2.png"

    p = argparse.ArgumentParser(description="ERA5 ocean conditioning (hard mask, v2: suppress predictions)")
    p.add_argument("--eval_parquet", type=str, default=str(default_eval))
    p.add_argument("--splits_json", type=str, default=str(default_splits))
    p.add_argument("--era5_daily_cache_dir", type=str, default=str(default_cache))

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use_weights", action="store_true")
    p.add_argument("--n_clusters", type=int, default=5)

    p.add_argument(
        "--selection_mode",
        type=str,
        default="grid_weather",
        choices=["grid_weather", "fixed_weather"],
        help="grid_weather=先对齐baseline FPR再选ERA5阈值；fixed_weather=固定ERA5阈值并在val上选分类阈值",
    )

    # 分类阈值校准（对齐 baseline FPR 量级）
    p.add_argument("--baseline_fpr_target", type=float, default=0.242, help="在 val 上选择分类阈值，使 baseline FPR 接近该目标")
    p.add_argument("--thr_grid_n", type=int, default=999)

    # ERA5 阈值搜索（val 上选择）
    p.add_argument("--conditioned_fpr_target", type=float, default=0.10, help="val 上 conditioned FPR 的目标上限")
    p.add_argument("--wind_grid", type=str, default="5,7.5,10,12.5,15", help="候选 wind 阈值网格（m/s），逗号分隔")
    p.add_argument("--precip_grid", type=str, default="1,2,5,10,20", help="候选 precip 阈值网格（mm/day），逗号分隔")

    # fixed_weather 模式参数
    p.add_argument("--fixed_wind_thresh_mps", type=float, default=10.0, help="fixed_weather: wind 阈值（m/s）")
    p.add_argument("--fixed_precip_thresh_mm_day", type=float, default=5.0, help="fixed_weather: precip 阈值（mm/day）")
    p.add_argument(
        "--thr_objective",
        type=str,
        default="mcc",
        choices=["mcc", "tp"],
        help="fixed_weather: 在满足 conditioned_fpr_target 前提下，阈值选择目标（mcc 或 tp）",
    )

    p.add_argument("--hgb_max_iter", type=int, default=200)
    p.add_argument("--hgb_max_depth", type=int, default=6)
    p.add_argument("--hgb_learning_rate", type=float, default=0.1)

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

    if bool(args.use_weights):
        train_df, (val_df, test_df), _ = add_foldwise_kmeans_weights(
            train_df,
            [val_df, test_df],
            feature_columns=FEATURE_COLUMNS,
            n_clusters=int(args.n_clusters),
            seed=int(args.seed),
        )

    x_tr, y_tr = _prepare_xy(train_df, use_weights=bool(args.use_weights))
    x_va, y_va = _prepare_xy(val_df, use_weights=bool(args.use_weights))
    x_te, y_te = _prepare_xy(test_df, use_weights=bool(args.use_weights))

    clf = HistGradientBoostingClassifier(
        max_iter=int(args.hgb_max_iter),
        max_depth=int(args.hgb_max_depth),
        learning_rate=float(args.hgb_learning_rate),
        random_state=int(args.seed),
    )
    clf.fit(x_tr, y_tr)

    p_va = clf.predict_proba(x_va)[:, 1].astype(np.float32)
    p_te = clf.predict_proba(x_te)[:, 1].astype(np.float32)

    # 预先附加 ERA5 covariates（val/test）
    val_cov = attach_era5_covariates(val_df[["sample_date", "grid_i", "grid_j"]].copy(), cache_dir=cache_dir)
    test_cov = attach_era5_covariates(test_df[["sample_date", "grid_i", "grid_j"]].copy(), cache_dir=cache_dir)
    wind_va = val_cov["era5_wind_speed"].to_numpy(np.float32)
    precip_va = val_cov["era5_precip_mm_day"].to_numpy(np.float32)
    wind_te = test_cov["era5_wind_speed"].to_numpy(np.float32)
    precip_te = test_cov["era5_precip_mm_day"].to_numpy(np.float32)

    wind_grid = None
    precip_grid = None

    selection_mode = str(args.selection_mode).strip().lower()
    if selection_mode == "grid_weather":
        thr_select_mode = "baseline_fpr_target"
        thr_objective = None

        thr_pred, fpr_val_baseline = _select_threshold_for_fpr_target(
            y_true=y_va,
            proba=p_va,
            fpr_target=float(args.baseline_fpr_target),
            grid_n=int(args.thr_grid_n),
        )
        logger.info(
            "val baseline: fpr_target=%.3f -> thr=%.4f (fpr_val=%.4f)",
            float(args.baseline_fpr_target),
            thr_pred,
            fpr_val_baseline,
        )

        yhat_va = (p_va >= float(thr_pred)).astype(np.int8)
        yhat_te = (p_te >= float(thr_pred)).astype(np.int8)

        # baseline metrics（无条件化）
        val_before = _binary_metrics(y_va, yhat_va)
        test_before = _binary_metrics(y_te, yhat_te)

        # 选择 weather thresholds（val 上满足 FPR<=target 且 MCC 最大；否则选 FPR 最小）
        wind_grid = _parse_grid(str(args.wind_grid))
        precip_grid = _parse_grid(str(args.precip_grid))

        best_feasible = None  # (mcc, -fpr, wt, pt, masked, metrics)
        best_any = None       # (fpr, -mcc, wt, pt, masked, metrics)

        for wt in wind_grid:
            for pt in precip_grid:
                keep_va = _compute_keep_mask(wind=wind_va, precip=precip_va, wind_thresh=wt, precip_thresh=pt)
                ys = _apply_weather_suppression(yhat_va, keep_va)
                m = _binary_metrics(y_va, ys)
                masked = int((~keep_va).sum())

                cand_any = (float(m.fpr), -float(m.mcc), float(wt), float(pt), masked, m)
                if best_any is None or cand_any < best_any:
                    best_any = cand_any

                if float(m.fpr) <= float(args.conditioned_fpr_target):
                    cand = (float(m.mcc), -float(m.fpr), float(wt), float(pt), masked, m)
                    if best_feasible is None or cand > best_feasible:
                        best_feasible = cand

        if best_feasible is not None:
            _, _, wind_sel, precip_sel, masked_val, val_after = best_feasible
            weather_select_mode = "feasible"
        else:
            assert best_any is not None
            _, _, wind_sel, precip_sel, masked_val, val_after = best_any
            weather_select_mode = "fallback_min_fpr"

        logger.info(
            "select weather (%s): wind=%.2f precip=%.2f masked_val=%d fpr_val_after=%.4f mcc_val_after=%.4f",
            weather_select_mode,
            float(wind_sel),
            float(precip_sel),
            int(masked_val),
            float(val_after.fpr),
            float(val_after.mcc),
        )

        keep_te = _compute_keep_mask(wind=wind_te, precip=precip_te, wind_thresh=float(wind_sel), precip_thresh=float(precip_sel))
        yhat_te_cond = _apply_weather_suppression(yhat_te, keep_te)
        test_after = _binary_metrics(y_te, yhat_te_cond)
    else:
        # fixed_weather
        wind_sel = float(args.fixed_wind_thresh_mps)
        precip_sel = float(args.fixed_precip_thresh_mm_day)
        keep_va = _compute_keep_mask(wind=wind_va, precip=precip_va, wind_thresh=wind_sel, precip_thresh=precip_sel)
        keep_te = _compute_keep_mask(wind=wind_te, precip=precip_te, wind_thresh=wind_sel, precip_thresh=precip_sel)

        thr_select_mode = "conditioned_fpr_target"
        thr_objective = str(args.thr_objective).strip().lower()
        thr_pred, thr_sel_info = _select_threshold_for_conditioned_fpr(
            y_true=y_va,
            proba=p_va,
            keep_mask=keep_va,
            fpr_max=float(args.conditioned_fpr_target),
            grid_n=int(args.thr_grid_n),
            objective=thr_objective,
        )
        weather_select_mode = "fixed"
        logger.info(
            "fixed weather: wind=%.2f precip=%.2f (masked_val=%d); select thr (%s/%s) -> %.4f",
            float(wind_sel),
            float(precip_sel),
            int((~keep_va).sum()),
            thr_sel_info.get("mode", ""),
            thr_sel_info.get("objective", ""),
            float(thr_pred),
        )

        yhat_va = (p_va >= float(thr_pred)).astype(np.int8)
        yhat_te = (p_te >= float(thr_pred)).astype(np.int8)
        val_before = _binary_metrics(y_va, yhat_va)
        test_before = _binary_metrics(y_te, yhat_te)

        yhat_va_cond = _apply_weather_suppression(yhat_va, keep_va)
        val_after = _binary_metrics(y_va, yhat_va_cond)

        yhat_te_cond = _apply_weather_suppression(yhat_te, keep_te)
        test_after = _binary_metrics(y_te, yhat_te_cond)

        masked_val = int((~keep_va).sum())
        fpr_val_baseline = float(val_before.fpr)

    # 输出表（以 test 为主，val 放入 json）
    pd.DataFrame(
        [
            {
                "subset": "zone_a_test",
                "model": "hist_gradient_boosting",
                "use_weights": bool(args.use_weights),
                "selection_mode": selection_mode,
                "thr_select_mode": thr_select_mode,
                "thr_select_objective": ("" if thr_objective is None else str(thr_objective)),
                "weather_select_mode": weather_select_mode,
                "baseline_fpr_target_val": (float(args.baseline_fpr_target) if selection_mode == "grid_weather" else float("nan")),
                "thr_pred_selected_on_val": float(thr_pred),
                "conditioned_fpr_target_val": float(args.conditioned_fpr_target),
                "wind_thresh_mps": float(wind_sel),
                "precip_thresh_mm_day": float(precip_sel),
                "n_total": int(test_before.n),
                "n_unmasked": int(np.sum(keep_te)),
                "n_masked": int(np.sum(~keep_te)),
                "fpr_before": float(test_before.fpr),
                "fpr_before_ci_low": float(test_before.fpr_ci_low),
                "fpr_before_ci_high": float(test_before.fpr_ci_high),
                "mcc_before": float(test_before.mcc),
                "fpr_after": float(test_after.fpr),
                "fpr_after_ci_low": float(test_after.fpr_ci_low),
                "fpr_after_ci_high": float(test_after.fpr_ci_high),
                "mcc_after": float(test_after.mcc),
            }
        ]
    ).to_csv(out_tab, index=False)

    # 图
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(5.8, 3.2))
        xs = ["Before", "After"]
        ys = [test_before.fpr, test_after.fpr]
        yerr_low = [max(0.0, test_before.fpr - test_before.fpr_ci_low), max(0.0, test_after.fpr - test_after.fpr_ci_low)]
        yerr_high = [max(0.0, test_before.fpr_ci_high - test_before.fpr), max(0.0, test_after.fpr_ci_high - test_after.fpr)]
        plt.bar(xs, ys, yerr=[yerr_low, yerr_high], capsize=5)
        plt.ylim(0, 1)
        plt.ylabel("FPR (Zone A test)")
        plt.title(f"ERA5 Screening (wind<={wind_sel:g} m/s, precip<={precip_sel:g} mm/day)")
        plt.tight_layout()
        plt.savefig(out_fig, dpi=200)
        plt.close()
    except Exception as e:
        logger.warning("绘图失败：%s", e)

    payload = {
        "version": "ocean_conditioning_v2",
        "generated_at_utc": datetime.utcnow().isoformat() + "Z",
        "inputs": {
            "eval_parquet": str(eval_parquet),
            "splits_json": str(splits_json),
            "era5_daily_cache_dir": str(cache_dir),
        },
        "params": {
            "seed": int(args.seed),
            "use_weights": bool(args.use_weights),
            "n_clusters": int(args.n_clusters),
            "selection_mode": selection_mode,
            "baseline_fpr_target": float(args.baseline_fpr_target),
            "thr_grid_n": int(args.thr_grid_n),
            "conditioned_fpr_target": float(args.conditioned_fpr_target),
            "wind_grid": (None if wind_grid is None else wind_grid),
            "precip_grid": (None if precip_grid is None else precip_grid),
            "fixed_wind_thresh_mps": float(args.fixed_wind_thresh_mps),
            "fixed_precip_thresh_mm_day": float(args.fixed_precip_thresh_mm_day),
            "thr_objective": str(args.thr_objective),
            "hgb_max_iter": int(args.hgb_max_iter),
            "hgb_max_depth": int(args.hgb_max_depth),
            "hgb_learning_rate": float(args.hgb_learning_rate),
        },
        "selection": {
            "thr_pred_selected_on_val": float(thr_pred),
            "val_fpr_baseline_at_thr": float(fpr_val_baseline),
            "thr_select_mode": thr_select_mode,
            "thr_select_objective": (None if thr_objective is None else str(thr_objective)),
            "weather_selection_mode": weather_select_mode,
            "wind_thresh_mps": float(wind_sel),
            "precip_thresh_mm_day": float(precip_sel),
        },
        "val": {
            "before": val_before.__dict__,
            "after": val_after.__dict__,
            "masked": int(masked_val),
        },
        "test": {
            "before": test_before.__dict__,
            "after": test_after.__dict__,
            "masked": int(np.sum(~keep_te)),
            "n_unmasked": int(np.sum(keep_te)),
        },
        "artifacts": {"table_csv": str(out_tab), "figure_png": str(out_fig)},
    }
    write_json(out_json, payload)
    logger.info("写入: %s", out_json)


if __name__ == "__main__":
    main()
