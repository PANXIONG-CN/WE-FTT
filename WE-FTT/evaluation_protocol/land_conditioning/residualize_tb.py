#!/usr/bin/env python3
"""
T7 陆地混杂残差化（Zone B–E，type=1..4）（v2）。

改进点：
- 阈值由 val 折选择（默认 max MCC），替代固定 0.5
- 背景残差模型支持按 zone 拟合（默认开启），并保留全局回退
- 残差协变量加入 ERA5（wind/precip + missing 指示），并保留 geo/season
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

weftt_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if weftt_root not in sys.path:
    sys.path.insert(0, weftt_root)

from evaluation_protocol.common.amsr2 import FEATURE_COLUMNS  # noqa: E402
from evaluation_protocol.common.jsonl import write_json  # noqa: E402
from evaluation_protocol.common.logging_utils import setup_logging  # noqa: E402
from evaluation_protocol.common.metrics import confusion_from_binary, fpr, matthews_corrcoef, wilson_ci  # noqa: E402
from evaluation_protocol.common.paths import get_repo_paths, resolve_path  # noqa: E402
from evaluation_protocol.common.splits import load_event_splits, split_df_by_event_id  # noqa: E402


logger = logging.getLogger(__name__)


LAND_ZONES = {1, 2, 3, 4}


def _doy(series_date: pd.Series) -> np.ndarray:
    d = pd.to_datetime(series_date, errors="coerce")
    return d.dt.dayofyear.astype("float32").to_numpy()


def _load_daily_cache(cache_dir: Path, prefix: str, ymd: str) -> np.ndarray:
    fp = cache_dir / f"{prefix}_{ymd}.npy"
    if not fp.exists():
        raise FileNotFoundError(fp)
    return np.load(fp)


def attach_era5_covariates(df: pd.DataFrame, *, cache_dir: Path) -> pd.DataFrame:
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
        logger.warning(
            "ERA5 daily cache 缺失 %d 天（例如：%s）；将保留 NaN 并由 missing 指示变量处理。",
            int(len(uniq)),
            head,
        )

    df["era5_wind_speed"] = ws_out
    df["era5_precip_mm_day"] = pr_out
    return df


def _build_covariates(
    df: pd.DataFrame,
    *,
    include_geo_season: bool,
    include_era5_covariates: bool,
) -> np.ndarray:
    cols: List[np.ndarray] = []

    if include_era5_covariates:
        ws = df["era5_wind_speed"].astype("float32").to_numpy()
        pr = df["era5_precip_mm_day"].astype("float32").to_numpy()
        ws_nan = np.isnan(ws).astype(np.float32)
        pr_nan = np.isnan(pr).astype(np.float32)
        ws = np.nan_to_num(ws, nan=0.0).astype(np.float32)
        pr = np.nan_to_num(pr, nan=0.0).astype(np.float32)
        cols.extend([ws, pr, ws_nan, pr_nan])

    if include_geo_season:
        lat = df["pixel_lat"].astype("float32").to_numpy()
        lon = df["pixel_lon"].astype("float32").to_numpy()
        doy = _doy(df["sample_date"])
        ang = 2.0 * np.pi * (doy / 365.25)
        cols.extend([lat, lon, np.sin(ang).astype(np.float32), np.cos(ang).astype(np.float32)])

    if not cols:
        raise ValueError("至少开启一种协变量：ERA5 或 geo/season。")
    return np.column_stack(cols).astype(np.float32)


def _fit_models_for_subset(
    subset_df: pd.DataFrame,
    *,
    alpha: float,
    include_geo_season: bool,
    include_era5_covariates: bool,
) -> Dict[str, Ridge]:
    x = _build_covariates(
        subset_df,
        include_geo_season=bool(include_geo_season),
        include_era5_covariates=bool(include_era5_covariates),
    )
    models: Dict[str, Ridge] = {}
    for col in FEATURE_COLUMNS:
        y = subset_df[col].astype("float32").to_numpy()
        m = Ridge(alpha=float(alpha), fit_intercept=True, random_state=42)
        m.fit(x, y)
        models[col] = m
    return models


def fit_background_models(
    train_controls: pd.DataFrame,
    *,
    alpha: float,
    include_geo_season: bool,
    include_era5_covariates: bool,
    zone_wise_background: bool,
) -> Dict[str, Dict[str, Ridge]]:
    models_by_zone: Dict[str, Dict[str, Ridge]] = {}
    models_by_zone["all"] = _fit_models_for_subset(
        train_controls,
        alpha=float(alpha),
        include_geo_season=bool(include_geo_season),
        include_era5_covariates=bool(include_era5_covariates),
    )

    if not zone_wise_background:
        return models_by_zone

    for z in sorted(LAND_ZONES):
        sub = train_controls[train_controls["zone_type"].astype(int) == int(z)].copy()
        if len(sub) < 100:
            logger.warning("zone=%d 控制样本过少（n=%d），回退全局背景模型。", int(z), int(len(sub)))
            continue
        models_by_zone[str(int(z))] = _fit_models_for_subset(
            sub,
            alpha=float(alpha),
            include_geo_season=bool(include_geo_season),
            include_era5_covariates=bool(include_era5_covariates),
        )
    return models_by_zone


def apply_residuals(
    df: pd.DataFrame,
    models_by_zone: Dict[str, Dict[str, Ridge]],
    *,
    include_geo_season: bool,
    include_era5_covariates: bool,
    suffix: str = "_resid",
) -> pd.DataFrame:
    if df.empty:
        return df

    if "all" not in models_by_zone:
        raise ValueError("models_by_zone 缺少全局模型键 all。")

    for col in FEATURE_COLUMNS:
        df[f"{col}{suffix}"] = np.nan

    for z, idx in df.groupby("zone_type").groups.items():
        zone_key = str(int(z))
        models = models_by_zone.get(zone_key, models_by_zone["all"])
        sub = df.loc[idx]
        x = _build_covariates(
            sub,
            include_geo_season=bool(include_geo_season),
            include_era5_covariates=bool(include_era5_covariates),
        )
        for col, model in models.items():
            y = sub[col].astype("float32").to_numpy()
            yhat = model.predict(x).astype(np.float32)
            df.loc[idx, f"{col}{suffix}"] = (y - yhat).astype(np.float32)
    return df


def train_classifier(train_df: pd.DataFrame, feature_cols: Sequence[str], *, seed: int) -> Pipeline:
    x = train_df[list(feature_cols)].astype("float32").to_numpy()
    y = train_df["flag"].astype("int64").to_numpy()
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


def _binary_metrics_at_threshold(y_true: np.ndarray, proba: np.ndarray, thr: float) -> Dict[str, float]:
    y_pred = (proba >= float(thr)).astype(np.int64)
    c = confusion_from_binary(y_true.astype(np.int64), y_pred)
    lo, hi = wilson_ci(successes=c.fp, n=(c.fp + c.tn))
    return {
        "thr": float(thr),
        "mcc": float(matthews_corrcoef(c)),
        "fpr": float(fpr(c)),
        "fpr_ci_low": float(lo),
        "fpr_ci_high": float(hi),
        "tp": int(c.tp),
        "fp": int(c.fp),
        "tn": int(c.tn),
        "fn": int(c.fn),
    }


def _select_threshold_on_val_max_mcc(*, y_true: np.ndarray, proba: np.ndarray, grid_n: int) -> Dict[str, float]:
    uniq = np.unique(proba.astype(np.float64))
    if uniq.size <= 2:
        grid = np.array([0.5], dtype=np.float64)
    else:
        grid = np.linspace(float(np.nanmin(uniq)), float(np.nanmax(uniq)), int(max(3, grid_n)))

    best = None
    for thr in grid:
        m = _binary_metrics_at_threshold(y_true, proba, float(thr))
        key = (float(m["mcc"]), -float(m["fpr"]), float(thr))
        if best is None or key > best[0]:
            best = (key, m)
    assert best is not None
    return best[1]


def eval_mcc_fpr(
    pipe: Pipeline,
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    thr: float,
) -> Dict[str, float]:
    x = df[list(feature_cols)].astype("float32").to_numpy()
    y_true = df["flag"].astype("int64").to_numpy()
    proba = pipe.predict_proba(x)[:, 1].astype(np.float64, copy=False)
    out = _binary_metrics_at_threshold(y_true, proba, float(thr))
    out["n"] = int(len(df))
    return out


def _placebo_control_vs_control_mcc(
    pipe: Pipeline,
    test_controls: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    thr: float,
    n_repeats: int,
    seed: int,
    min_controls_per_event: int,
) -> List[float]:
    rng = np.random.default_rng(int(seed))
    g = test_controls.groupby("event_id")
    events = []
    for eid, df_e in g:
        cidx = sorted(set(df_e["control_index"].astype(int).tolist()))
        cidx = [x for x in cidx if x >= 0]
        if len(cidx) >= int(min_controls_per_event):
            events.append((str(eid), cidx))
    if not events:
        raise ValueError("test 折无法构造 placebo：每个事件至少需要 2 个 control_index。")

    out: List[float] = []
    for _ in range(int(n_repeats)):
        parts = []
        y_parts = []
        for eid, cidx in events:
            a, b = rng.choice(cidx, size=2, replace=False)
            df_a = test_controls[(test_controls["event_id"] == eid) & (test_controls["control_index"] == int(a))]
            df_b = test_controls[(test_controls["event_id"] == eid) & (test_controls["control_index"] == int(b))]
            if df_a.empty or df_b.empty:
                continue
            parts.append(df_a)
            y_parts.append(np.ones(len(df_a), dtype=np.int64))
            parts.append(df_b)
            y_parts.append(np.zeros(len(df_b), dtype=np.int64))

        if not parts:
            out.append(0.0)
            continue

        df_p = pd.concat(parts, ignore_index=True)
        y_true = np.concatenate(y_parts, axis=0)
        x = df_p[list(feature_cols)].astype("float32").to_numpy()
        proba = pipe.predict_proba(x)[:, 1].astype(np.float64, copy=False)
        y_pred = (proba >= float(thr)).astype(np.int64)
        c = confusion_from_binary(y_true, y_pred)
        out.append(float(matthews_corrcoef(c)))
    return out


def parse_args():
    repo = get_repo_paths()
    default_eval = repo.eval_root / "datasets" / "mbt_eval_samples_v1.parquet"
    default_splits = repo.eval_root / "data_splits" / "event_grouped_splits_v1.json"
    default_cache = repo.eval_root / "era5" / "cache" / "daily"
    default_out = repo.eval_root / "land_conditioning" / "results" / "land_residualize_v2.json"
    default_tab = repo.eval_root / "land_conditioning" / "tables" / "tab_s9_land_residualize_v2.csv"
    default_fig = repo.eval_root / "land_conditioning" / "figures" / "fig_s7_land_residualize_v2.png"

    p = argparse.ArgumentParser(description="Land residualization (v2)")
    p.add_argument("--eval_parquet", type=str, default=str(default_eval))
    p.add_argument("--splits_json", type=str, default=str(default_splits))
    p.add_argument("--era5_daily_cache_dir", type=str, default=str(default_cache))
    p.add_argument("--alpha", type=float, default=1.0, help="Ridge 正则强度")
    p.add_argument("--include_geo_season", action="store_true", help="协变量包含 lat/lon 与 sin/cos(DOY)")
    p.add_argument("--no_era5_covariates", action="store_false", dest="include_era5_covariates", help="关闭 ERA5 协变量")
    p.add_argument("--no_zone_wise_background", action="store_false", dest="zone_wise_background", help="关闭按 zone 背景拟合")
    p.add_argument("--thr_select_mode", type=str, default="val_mcc", choices=["val_mcc", "fixed"])
    p.add_argument("--thr_fixed", type=float, default=0.5)
    p.add_argument("--thr_grid_n", type=int, default=999)
    p.add_argument("--n_placebo_repeats", type=int, default=100)
    p.add_argument("--min_controls_per_event", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_json", type=str, default=str(default_out))
    p.add_argument("--out_tab_csv", type=str, default=str(default_tab))
    p.add_argument("--out_fig_png", type=str, default=str(default_fig))
    p.add_argument("--log_file", type=str, default=None)
    p.set_defaults(include_era5_covariates=True, zone_wise_background=True)
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging(log_file=args.log_file)

    eval_parquet = resolve_path(args.eval_parquet)
    splits_json = resolve_path(args.splits_json)
    era5_cache_dir = resolve_path(args.era5_daily_cache_dir)
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
        "control_index",
        "sample_date",
        "pixel_lat",
        "pixel_lon",
        "grid_i",
        "grid_j",
        *FEATURE_COLUMNS,
    ]
    df = pd.read_parquet(eval_parquet, columns=need_cols)
    df = df[df["zone_type"].isin(LAND_ZONES)].copy()
    if df.empty:
        raise ValueError("eval_parquet 中未找到陆地区（zone_type=1..4）样本。")

    if bool(args.include_era5_covariates):
        df = attach_era5_covariates(df, cache_dir=era5_cache_dir)

    splits = load_event_splits(splits_json)
    train_df, val_df, test_df = split_df_by_event_id(df, splits)

    train_bg = train_df[(train_df["flag"].astype(int) == 0) & (train_df["window_type"].astype(str) == "control")].copy()
    if train_bg.empty:
        raise ValueError("训练折未包含对照窗样本，无法拟合背景模型。")

    bg_models = fit_background_models(
        train_bg,
        alpha=float(args.alpha),
        include_geo_season=bool(args.include_geo_season),
        include_era5_covariates=bool(args.include_era5_covariates),
        zone_wise_background=bool(args.zone_wise_background),
    )

    train_df = apply_residuals(
        train_df,
        bg_models,
        include_geo_season=bool(args.include_geo_season),
        include_era5_covariates=bool(args.include_era5_covariates),
    )
    val_df = apply_residuals(
        val_df,
        bg_models,
        include_geo_season=bool(args.include_geo_season),
        include_era5_covariates=bool(args.include_era5_covariates),
    )
    test_df = apply_residuals(
        test_df,
        bg_models,
        include_geo_season=bool(args.include_geo_season),
        include_era5_covariates=bool(args.include_era5_covariates),
    )

    resid_cols = [f"{c}_resid" for c in FEATURE_COLUMNS]
    pipe_raw = train_classifier(train_df, FEATURE_COLUMNS, seed=int(args.seed))
    pipe_resid = train_classifier(train_df, resid_cols, seed=int(args.seed))

    x_val_raw = val_df[list(FEATURE_COLUMNS)].astype("float32").to_numpy()
    x_val_resid = val_df[list(resid_cols)].astype("float32").to_numpy()
    y_val = val_df["flag"].astype("int64").to_numpy()
    p_val_raw = pipe_raw.predict_proba(x_val_raw)[:, 1].astype(np.float64, copy=False)
    p_val_resid = pipe_resid.predict_proba(x_val_resid)[:, 1].astype(np.float64, copy=False)

    thr_mode = str(args.thr_select_mode).strip().lower()
    if thr_mode == "fixed":
        thr_raw = float(args.thr_fixed)
        thr_resid = float(args.thr_fixed)
        val_sel_raw = _binary_metrics_at_threshold(y_val, p_val_raw, thr_raw)
        val_sel_resid = _binary_metrics_at_threshold(y_val, p_val_resid, thr_resid)
    else:
        val_sel_raw = _select_threshold_on_val_max_mcc(y_true=y_val, proba=p_val_raw, grid_n=int(args.thr_grid_n))
        val_sel_resid = _select_threshold_on_val_max_mcc(y_true=y_val, proba=p_val_resid, grid_n=int(args.thr_grid_n))
        thr_raw = float(val_sel_raw["thr"])
        thr_resid = float(val_sel_resid["thr"])

    real_raw = eval_mcc_fpr(pipe_raw, test_df, FEATURE_COLUMNS, thr=float(thr_raw))
    real_resid = eval_mcc_fpr(pipe_resid, test_df, resid_cols, thr=float(thr_resid))

    test_controls = test_df[(test_df["flag"].astype(int) == 0) & (test_df["window_type"].astype(str) == "control")].copy()

    def summarize(placebo: List[float], real_mcc: float):
        arr = np.asarray(placebo, dtype=np.float64)
        mu = float(np.mean(arr)) if arr.size else 0.0
        sd = float(np.std(arr, ddof=1)) if arr.size >= 2 else 0.0
        z = (float(real_mcc) - mu) / (sd + 1e-12)
        p = (1.0 + float(np.sum(arr >= float(real_mcc)))) / (float(arr.size) + 1.0)
        return {"mcc_values": [float(x) for x in placebo], "mean": mu, "std": sd, "z_score": float(z), "p_value_ge": float(p)}

    by_zone: Dict[str, Dict] = {}
    table_rows: List[Dict] = []

    def add_zone_row(name: str, subset_test: pd.DataFrame, subset_controls: pd.DataFrame):
        rr = eval_mcc_fpr(pipe_raw, subset_test, FEATURE_COLUMNS, thr=float(thr_raw))
        rres = eval_mcc_fpr(pipe_resid, subset_test, resid_cols, thr=float(thr_resid))
        pr = _placebo_control_vs_control_mcc(
            pipe_raw,
            subset_controls,
            FEATURE_COLUMNS,
            thr=float(thr_raw),
            n_repeats=int(args.n_placebo_repeats),
            seed=int(args.seed) + 101,
            min_controls_per_event=int(args.min_controls_per_event),
        )
        pres = _placebo_control_vs_control_mcc(
            pipe_resid,
            subset_controls,
            resid_cols,
            thr=float(thr_resid),
            n_repeats=int(args.n_placebo_repeats),
            seed=int(args.seed) + 202,
            min_controls_per_event=int(args.min_controls_per_event),
        )
        sr = summarize(pr, rr["mcc"])
        sres = summarize(pres, rres["mcc"])
        by_zone[name] = {"real": {"raw": rr, "residual": rres}, "placebo": {"raw": sr, "residual": sres}}
        table_rows.append(
            {
                "subset": name,
                "thr_raw_selected_on_val": float(thr_raw),
                "thr_resid_selected_on_val": float(thr_resid),
                "real_mcc_raw": float(rr["mcc"]),
                "placebo_mean_raw": float(sr["mean"]),
                "placebo_std_raw": float(sr["std"]),
                "p_value_raw_ge": float(sr["p_value_ge"]),
                "real_mcc_resid": float(rres["mcc"]),
                "placebo_mean_resid": float(sres["mean"]),
                "placebo_std_resid": float(sres["std"]),
                "p_value_resid_ge": float(sres["p_value_ge"]),
            }
        )

    add_zone_row("land_all", test_df, test_controls)
    for z in sorted(LAND_ZONES):
        sub_test = test_df[test_df["zone_type"] == z]
        sub_ctrl = test_controls[test_controls["zone_type"] == z]
        if sub_test.empty or sub_ctrl.empty:
            continue
        add_zone_row(f"zone_{int(z)}", sub_test, sub_ctrl)

    land_all_placebo = by_zone.get("land_all", {}).get("placebo", {})
    payload = {
        "version": "land_residualize_v2",
        "generated_at_utc": datetime.utcnow().isoformat() + "Z",
        "inputs": {
            "eval_parquet": str(eval_parquet),
            "splits_json": str(splits_json),
            "era5_daily_cache_dir": (str(era5_cache_dir) if bool(args.include_era5_covariates) else None),
        },
        "params": {
            "alpha": float(args.alpha),
            "n_placebo_repeats": int(args.n_placebo_repeats),
            "min_controls_per_event": int(args.min_controls_per_event),
            "seed": int(args.seed),
            "zones": sorted(list(LAND_ZONES)),
            "thr_select_mode": str(thr_mode),
            "thr_fixed": float(args.thr_fixed),
            "thr_grid_n": int(args.thr_grid_n),
            "include_geo_season": bool(args.include_geo_season),
            "include_era5_covariates": bool(args.include_era5_covariates),
            "zone_wise_background": bool(args.zone_wise_background),
            "zone_models_available": sorted(k for k in bg_models.keys() if k != "all"),
        },
        "selection": {
            "thr_raw_selected_on_val": float(thr_raw),
            "thr_resid_selected_on_val": float(thr_resid),
            "val_raw_at_thr": val_sel_raw,
            "val_resid_at_thr": val_sel_resid,
        },
        "real": {"raw": real_raw, "residual": real_resid},
        "placebo": land_all_placebo,
        "by_zone": by_zone,
        "artifacts": {"table_csv": str(out_tab), "figure_png": str(out_fig)},
    }
    write_json(out_json, payload)

    pd.DataFrame(table_rows).to_csv(out_tab, index=False)

    try:
        import matplotlib.pyplot as plt

        land_all = by_zone.get("land_all", {})
        placebo_raw_vals = land_all.get("placebo", {}).get("raw", {}).get("mcc_values", [])
        placebo_resid_vals = land_all.get("placebo", {}).get("residual", {}).get("mcc_values", [])

        plt.figure(figsize=(7.2, 3.2))
        data = [placebo_raw_vals, placebo_resid_vals]
        plt.boxplot(data, tick_labels=["Raw", "Residual"], showfliers=False)
        plt.axhline(float(real_raw["mcc"]), color="red", linewidth=2, label="Real (Raw)")
        plt.axhline(float(real_resid["mcc"]), color="green", linewidth=2, label="Real (Residual)")
        plt.ylabel("MCC")
        plt.title("Land (Zone B–E): Real vs Placebo (control vs control)")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(out_fig, dpi=200)
        plt.close()
    except Exception as e:
        logger.warning("绘图失败：%s", e)

    logger.info("写入: %s", out_json)


if __name__ == "__main__":
    main()
