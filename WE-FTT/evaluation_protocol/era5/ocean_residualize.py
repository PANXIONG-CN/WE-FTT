#!/usr/bin/env python3
"""
T6 海洋区（Zone A / type=0）ERA5 条件化：残差化路线（v1，最小可用实现）。

核心想法：
- 使用训练折内的“对照窗样本”（flag=0, window_type=control）拟合背景模型：TB ~ ERA5(wind, precip) (+ 可选 geo/season)
- 用残差 TB 作为新特征训练同一个分类器，并在 test 折评估 MCC/FPR（含 Wilson CI）

与硬掩膜（路线1）不同：残差化不会丢弃样本，而是显式剔除 ERA5 可解释的 TB 变化。
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, SGDClassifier
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
from evaluation_protocol.common.weighting import add_foldwise_kmeans_weights, artifacts_to_jsonable  # noqa: E402


logger = logging.getLogger(__name__)


def _doy(series_date: pd.Series) -> np.ndarray:
    d = pd.to_datetime(series_date, errors="coerce")
    return d.dt.dayofyear.astype("float32").to_numpy()


def _load_daily_cache(cache_dir: Path, prefix: str, ymd: str) -> np.ndarray:
    fp = cache_dir / f"{prefix}_{ymd}.npy"
    if not fp.exists():
        raise FileNotFoundError(fp)
    return np.load(fp)


def attach_era5_covariates(df: pd.DataFrame, *, cache_dir: Path) -> pd.DataFrame:
    """
    为每行样本按 (sample_date, grid_i, grid_j) 取 ERA5 值，写入：
    - era5_wind_speed（m/s）
    - era5_precip_mm_day（mm/day）
    """
    if df.empty:
        return df

    df = df.reset_index(drop=True)
    ws_out = np.full(len(df), np.nan, dtype=np.float32)
    pr_out = np.full(len(df), np.nan, dtype=np.float32)

    missing_days: List[str] = []
    for d, idx in df.groupby("sample_date").groups.items():
        ymd = str(d).replace("-", "")
        try:
            ws = _load_daily_cache(cache_dir, "wind_speed", ymd)
            pr = _load_daily_cache(cache_dir, "precip_mm", ymd)
        except FileNotFoundError:
            missing_days.append(ymd)
            continue

        ii = df.loc[idx, "grid_i"].astype(int).to_numpy()
        jj = df.loc[idx, "grid_j"].astype(int).to_numpy()
        ws_out[idx] = ws[ii, jj].astype(np.float32)
        pr_out[idx] = pr[ii, jj].astype(np.float32)

    if missing_days:
        uniq = sorted(set(missing_days))
        head = ", ".join(uniq[:10])
        raise FileNotFoundError(
            f"ERA5 daily cache 缺失 {len(uniq)} 天（例如：{head}）。"
            "请先运行 download_from_cds.py 补齐所需日期（建议 --use_split all）。"
        )

    df["era5_wind_speed"] = ws_out
    df["era5_precip_mm_day"] = pr_out
    return df


def _build_covariates(df: pd.DataFrame, *, include_geo_season: bool) -> np.ndarray:
    ws = df["era5_wind_speed"].astype("float32").to_numpy()
    pr = df["era5_precip_mm_day"].astype("float32").to_numpy()
    ws_nan = np.isnan(ws).astype(np.float32)
    pr_nan = np.isnan(pr).astype(np.float32)
    ws = np.nan_to_num(ws, nan=0.0).astype(np.float32)
    pr = np.nan_to_num(pr, nan=0.0).astype(np.float32)

    cols = [ws, pr, ws_nan, pr_nan]

    if include_geo_season:
        lat = df["pixel_lat"].astype("float32").to_numpy()
        lon = df["pixel_lon"].astype("float32").to_numpy()
        doy = _doy(df["sample_date"])
        ang = 2.0 * np.pi * (doy / 365.25)
        cols.extend([lat, lon, np.sin(ang).astype(np.float32), np.cos(ang).astype(np.float32)])

    return np.column_stack(cols).astype(np.float32)


def fit_background_models(train_controls: pd.DataFrame, *, alpha: float, include_geo_season: bool) -> Dict[str, Ridge]:
    x = _build_covariates(train_controls, include_geo_season=bool(include_geo_season))
    models: Dict[str, Ridge] = {}
    for col in FEATURE_COLUMNS:
        y = train_controls[col].astype("float32").to_numpy()
        m = Ridge(alpha=float(alpha), fit_intercept=True, random_state=42)
        m.fit(x, y)
        models[col] = m
    return models


def apply_residuals(df: pd.DataFrame, models: Dict[str, Ridge], *, include_geo_season: bool, suffix: str = "_resid") -> pd.DataFrame:
    if df.empty:
        return df
    x = _build_covariates(df, include_geo_season=bool(include_geo_season))
    for col, m in models.items():
        y = df[col].astype("float32").to_numpy()
        yhat = m.predict(x).astype(np.float32)
        df[f"{col}{suffix}"] = (y - yhat).astype(np.float32)
    return df


def _prepare_xy(df: pd.DataFrame, feature_cols: Sequence[str], *, use_weights: bool) -> Tuple[np.ndarray, np.ndarray]:
    y = df["flag"].astype("int64").to_numpy()
    x_feat = df[list(feature_cols)].astype("float32").to_numpy()
    if not use_weights:
        return x_feat, y
    w_cols = [f"{c}_cluster_labels_weight" for c in feature_cols]
    x_w = df[w_cols].astype("float32").to_numpy()
    return (x_feat * x_w), y


def train_classifier(train_df: pd.DataFrame, feature_cols: Sequence[str], *, use_weights: bool, seed: int) -> Pipeline:
    x, y = _prepare_xy(train_df, feature_cols, use_weights=bool(use_weights))
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


def eval_metrics(pipe: Pipeline, df: pd.DataFrame, feature_cols: Sequence[str], *, use_weights: bool) -> Dict[str, float]:
    x, y_true = _prepare_xy(df, feature_cols, use_weights=bool(use_weights))
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


def parse_args():
    repo = get_repo_paths()
    default_eval = repo.eval_root / "datasets" / "mbt_eval_samples_v1.parquet"
    default_splits = repo.eval_root / "data_splits" / "event_grouped_splits_v1.json"
    default_cache = repo.eval_root / "era5" / "cache" / "daily"
    default_out = repo.eval_root / "era5" / "results" / "ocean_residualize_v1.json"
    default_tab = repo.eval_root / "era5" / "tables" / "tab_s7_era5_ocean_residualize_v1.csv"
    default_fig = repo.eval_root / "era5" / "figures" / "fig_s6_era5_ocean_residualize_v1.png"

    p = argparse.ArgumentParser(description="ERA5 ocean conditioning (residualization, v1)")
    p.add_argument("--eval_parquet", type=str, default=str(default_eval))
    p.add_argument("--splits_json", type=str, default=str(default_splits))
    p.add_argument("--era5_daily_cache_dir", type=str, default=str(default_cache))
    p.add_argument("--alpha", type=float, default=1.0, help="Ridge 正则强度")
    p.add_argument("--include_geo_season", action="store_true", help="额外加入 lat/lon 与 sin/cos(DOY) 作为协变量")
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

    need_cols = [
        "event_id",
        "zone_type",
        "flag",
        "window_type",
        "sample_date",
        "grid_i",
        "grid_j",
        "pixel_lat",
        "pixel_lon",
        *FEATURE_COLUMNS,
    ]
    df = pd.read_parquet(eval_parquet, columns=need_cols)
    df = df[df["zone_type"].astype(int) == 0].copy()
    if df.empty:
        raise ValueError("eval_parquet 中未找到 Zone A（zone_type=0）样本。")

    df = attach_era5_covariates(df, cache_dir=cache_dir)

    splits = load_event_splits(splits_json)
    train_df, val_df, test_df = split_df_by_event_id(df, splits)

    train_controls = train_df[(train_df["flag"].astype(int) == 0) & (train_df["window_type"].astype(str) == "control")].copy()
    if train_controls.empty:
        raise ValueError("训练折未包含对照窗样本，无法拟合背景模型。")

    bg_models = fit_background_models(
        train_controls,
        alpha=float(args.alpha),
        include_geo_season=bool(args.include_geo_season),
    )

    train_df = apply_residuals(train_df, bg_models, include_geo_season=bool(args.include_geo_season))
    val_df = apply_residuals(val_df, bg_models, include_geo_season=bool(args.include_geo_season))
    test_df = apply_residuals(test_df, bg_models, include_geo_season=bool(args.include_geo_season))

    resid_cols = [f"{c}_resid" for c in FEATURE_COLUMNS]

    artifacts_raw = []
    artifacts_resid = []
    if bool(args.use_weights):
        train_df, (val_df, test_df), artifacts_raw = add_foldwise_kmeans_weights(
            train_df,
            [val_df, test_df],
            feature_columns=FEATURE_COLUMNS,
            n_clusters=int(args.n_clusters),
            seed=int(args.seed),
        )
        train_df, (val_df, test_df), artifacts_resid = add_foldwise_kmeans_weights(
            train_df,
            [val_df, test_df],
            feature_columns=resid_cols,
            n_clusters=int(args.n_clusters),
            seed=int(args.seed) + 7,
        )

    pipe_raw = train_classifier(train_df, FEATURE_COLUMNS, use_weights=bool(args.use_weights), seed=int(args.seed))
    pipe_resid = train_classifier(train_df, resid_cols, use_weights=bool(args.use_weights), seed=int(args.seed))

    raw = eval_metrics(pipe_raw, test_df, FEATURE_COLUMNS, use_weights=bool(args.use_weights))
    resid = eval_metrics(pipe_resid, test_df, resid_cols, use_weights=bool(args.use_weights))

    pd.DataFrame(
        [
            {
                "subset": "zone_a_test",
                "alpha": float(args.alpha),
                "include_geo_season": bool(args.include_geo_season),
                "use_weights": bool(args.use_weights),
                "n": int(raw["n"]),
                "fpr_raw": float(raw["fpr"]),
                "fpr_raw_ci_low": float(raw["fpr_ci_low"]),
                "fpr_raw_ci_high": float(raw["fpr_ci_high"]),
                "mcc_raw": float(raw["mcc"]),
                "fpr_resid": float(resid["fpr"]),
                "fpr_resid_ci_low": float(resid["fpr_ci_low"]),
                "fpr_resid_ci_high": float(resid["fpr_ci_high"]),
                "mcc_resid": float(resid["mcc"]),
            }
        ]
    ).to_csv(out_tab, index=False)

    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6.0, 3.2))
        xs = ["Raw", "Residual"]
        ys = [raw["fpr"], resid["fpr"]]
        yerr = [
            [raw["fpr"] - raw["fpr_ci_low"], resid["fpr"] - resid["fpr_ci_low"]],
            [raw["fpr_ci_high"] - raw["fpr"], resid["fpr_ci_high"] - resid["fpr"]],
        ]
        plt.bar(xs, ys, yerr=yerr, capsize=5)
        plt.ylim(0, 1)
        plt.ylabel("FPR (Zone A test)")
        plt.title("ERA5 Residualization Conditioning")
        plt.tight_layout()
        plt.savefig(out_fig, dpi=200)
        plt.close()
    except Exception as e:
        logger.warning("绘图失败：%s", e)

    model_dump = {}
    for col, m in bg_models.items():
        model_dump[str(col)] = {"intercept": float(m.intercept_), "coef": [float(x) for x in np.asarray(m.coef_).reshape(-1)]}

    payload = {
        "version": "ocean_residualize_v1",
        "generated_at_utc": datetime.utcnow().isoformat() + "Z",
        "inputs": {
            "eval_parquet": str(eval_parquet),
            "splits_json": str(splits_json),
            "era5_daily_cache_dir": str(cache_dir),
        },
        "params": {
            "alpha": float(args.alpha),
            "include_geo_season": bool(args.include_geo_season),
            "seed": int(args.seed),
            "use_weights": bool(args.use_weights),
            "n_clusters": int(args.n_clusters),
        },
        "raw": raw,
        "residual": resid,
        "background_models": model_dump,
        "weighting": {
            "raw": artifacts_to_jsonable(artifacts_raw),
            "residual": artifacts_to_jsonable(artifacts_resid),
        },
        "artifacts": {"table_csv": str(out_tab), "figure_png": str(out_fig)},
    }
    write_json(out_json, payload)
    logger.info("写入: %s", out_json)


if __name__ == "__main__":
    main()

