#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

weftt_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if weftt_root not in sys.path:
    sys.path.insert(0, weftt_root)

from evaluation_protocol.common.amsr2 import FEATURE_COLUMNS  # noqa: E402
from evaluation_protocol.common.metrics import confusion_from_binary, fpr, matthews_corrcoef, wilson_ci  # noqa: E402
from evaluation_protocol.common.paths import get_repo_paths, resolve_path  # noqa: E402
from evaluation_protocol.common.splits import load_event_splits, split_df_by_event_id  # noqa: E402
from evaluation_protocol.common.weighting import add_foldwise_kmeans_weights  # noqa: E402


def parse_args():
    repo = get_repo_paths()
    default_eval = repo.eval_root / "datasets" / "mbt_eval_samples_v1.parquet"
    default_splits = repo.eval_root / "data_splits" / "event_grouped_splits_v1.json"
    default_era5_cache = repo.eval_root / "era5" / "cache" / "daily"
    default_csv = repo.eval_root / "data_splits" / "tables" / "tab_s11_spatial_cv_blocks.csv"
    default_json = repo.eval_root / "data_splits" / "results" / "spatial_cv_summary_v1.json"
    default_fig = repo.eval_root / "data_splits" / "figures" / "fig_s8_spatial_cv_mcc.png"

    p = argparse.ArgumentParser(description="10x10 block leave-one-out spatial CV (event-grouped).")
    p.add_argument("--eval_parquet", type=str, default=str(default_eval))
    p.add_argument("--splits_json", type=str, default=str(default_splits))
    p.add_argument("--aggregation", type=str, default="pixel", choices=["pixel", "day_mean"])
    p.add_argument("--block_size_deg", type=float, default=10.0)
    p.add_argument("--min_events_per_block", type=int, default=2)
    p.add_argument("--val_ratio", type=float, default=0.10)
    p.add_argument("--max_folds", type=int, default=None)
    p.add_argument("--min_train_rows_per_fold", type=int, default=5000)
    p.add_argument("--min_val_rows_per_fold", type=int, default=1000)
    p.add_argument("--min_test_rows_per_fold", type=int, default=1000)
    p.add_argument("--failure_mcc", type=float, default=0.60)
    p.add_argument("--max_train_rows", type=int, default=400000)
    p.add_argument("--use_weights", action="store_true")
    p.add_argument("--n_clusters", type=int, default=5)
    p.add_argument("--model", type=str, default="sgd", choices=["sgd", "hgb"])
    p.add_argument("--zone_wise", action="store_true")
    p.add_argument("--min_zone_train_rows", type=int, default=5000)
    p.add_argument("--zone_adaptive_thresholds", action="store_true")
    p.add_argument("--min_zone_val_rows_for_threshold", type=int, default=300)
    p.add_argument("--min_zone_test_rows_for_threshold", type=int, default=300)
    p.add_argument("--hgb_max_iter", type=int, default=300)
    p.add_argument("--hgb_max_depth", type=int, default=6)
    p.add_argument("--hgb_learning_rate", type=float, default=0.06)
    p.add_argument("--thr_grid_n", type=int, default=301)
    p.add_argument("--threshold_objective", type=str, default="mcc", choices=["mcc", "fpr_target"])
    p.add_argument("--fpr_target", type=float, default=0.15)
    p.add_argument("--land_residualize", action="store_true")
    p.add_argument("--era5_daily_cache_dir", type=str, default=str(default_era5_cache))
    p.add_argument("--land_resid_alpha", type=float, default=1.0)
    p.add_argument("--land_resid_include_geo_season", action="store_true")
    p.add_argument("--land_resid_include_era5_covariates", action="store_true")
    p.add_argument("--land_resid_zone_wise_background", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_csv", type=str, default=str(default_csv))
    p.add_argument("--out_json", type=str, default=str(default_json))
    p.add_argument("--out_fig", type=str, default=str(default_fig))
    return p.parse_args()


def _block_key(lat: float, lon: float, block_size_deg: float) -> str:
    lat_idx = int(np.floor((float(lat) + 90.0) / float(block_size_deg)))
    lon_idx = int(np.floor((float(lon) + 180.0) / float(block_size_deg)))
    n_lat = int(np.ceil(180.0 / float(block_size_deg)))
    n_lon = int(np.ceil(360.0 / float(block_size_deg)))
    lat_idx = int(np.clip(lat_idx, 0, max(0, n_lat - 1)))
    lon_idx = int(np.clip(lon_idx, 0, max(0, n_lon - 1)))
    lat0 = -90 + lat_idx * int(block_size_deg)
    lon0 = -180 + lon_idx * int(block_size_deg)
    return f"{lat0:+03d}_{lon0:+04d}"


def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    c = confusion_from_binary(y_true.astype(np.int64), y_pred.astype(np.int64))
    lo, hi = wilson_ci(successes=int(c.fp), n=int(c.fp + c.tn))
    return {
        "mcc": float(matthews_corrcoef(c)),
        "fpr": float(fpr(c)),
        "fpr_ci_low": float(lo),
        "fpr_ci_high": float(hi),
        "tp": int(c.tp),
        "fp": int(c.fp),
        "tn": int(c.tn),
        "fn": int(c.fn),
        "n": int(len(y_true)),
    }


def _feature_matrix(df: pd.DataFrame, feature_columns: Sequence[str], use_weights: bool) -> np.ndarray:
    feature_columns = list(feature_columns)
    x = df[feature_columns].astype("float32").to_numpy()
    if bool(use_weights):
        w_cols = [f"{c}_cluster_labels_weight" for c in feature_columns]
        if all(c in df.columns for c in w_cols):
            w = df[w_cols].astype("float32").to_numpy()
            x = x * w
    return x


def _select_threshold(
    y_true: np.ndarray,
    proba: np.ndarray,
    grid_n: int,
    objective: str,
    fpr_target: float,
) -> Tuple[float, str]:
    grid = np.linspace(0.001, 0.999, int(max(31, grid_n)))
    objective = str(objective).strip().lower()
    if objective == "mcc":
        best = None
        for thr in grid:
            m = _binary_metrics(y_true, (proba >= float(thr)).astype(np.int64))
            key = (float(m["mcc"]), -float(m["fpr"]), float(thr))
            if best is None or key > best[0]:
                best = (key, float(thr))
        assert best is not None
        return float(best[1]), "mcc"

    feasible = None
    fallback = None
    for thr in grid:
        m = _binary_metrics(y_true, (proba >= float(thr)).astype(np.int64))
        fpr_v = float(m["fpr"])
        mcc_v = float(m["mcc"])
        fb_key = (-fpr_v, mcc_v, float(thr))
        if fallback is None or fb_key > fallback[0]:
            fallback = (fb_key, float(thr))
        if fpr_v <= float(fpr_target):
            key = (mcc_v, -abs(fpr_v - float(fpr_target)), float(thr))
            if feasible is None or key > feasible[0]:
                feasible = (key, float(thr))
    if feasible is not None:
        return float(feasible[1]), "fpr_target_feasible"
    assert fallback is not None
    return float(fallback[1]), "fpr_target_fallback"


def _build_model(args, seed: int):
    if str(args.model) == "hgb":
        return HistGradientBoostingClassifier(
            max_depth=int(args.hgb_max_depth),
            max_iter=int(args.hgb_max_iter),
            learning_rate=float(args.hgb_learning_rate),
            random_state=int(seed),
        )
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                SGDClassifier(
                    loss="log_loss",
                    alpha=1e-4,
                    max_iter=3000,
                    tol=1e-3,
                    class_weight="balanced",
                    random_state=int(seed),
                ),
            ),
        ]
    )


def _fit_single(train_df: pd.DataFrame, args, seed: int, feature_columns: Sequence[str]):
    x = _feature_matrix(train_df, feature_columns, bool(args.use_weights))
    y = train_df["flag"].astype("int64").to_numpy()
    model = _build_model(args, seed=int(seed))
    model.fit(x, y)
    return model


def _fit_model_bundle(train_df: pd.DataFrame, args, seed: int, feature_columns: Sequence[str]):
    global_model = _fit_single(train_df, args, seed=int(seed), feature_columns=feature_columns)
    zone_models: Dict[int, object] = {}
    if bool(args.zone_wise):
        for z, sub in train_df.groupby("zone_type"):
            sub = sub.copy()
            if len(sub) < int(args.min_zone_train_rows):
                continue
            y = sub["flag"].astype("int64").to_numpy()
            if len(np.unique(y)) < 2:
                continue
            zone_models[int(z)] = _fit_single(
                sub,
                args,
                seed=int(seed) + int(z) * 31,
                feature_columns=feature_columns,
            )
    return global_model, zone_models


def _predict_bundle(df: pd.DataFrame, bundle, args, feature_columns: Sequence[str]) -> np.ndarray:
    global_model, zone_models = bundle
    x = _feature_matrix(df, feature_columns, bool(args.use_weights))
    if not zone_models:
        return global_model.predict_proba(x)[:, 1].astype(np.float64, copy=False)
    zones = df["zone_type"].astype(int).to_numpy()
    out = np.empty(len(df), dtype=np.float64)
    assigned = np.zeros(len(df), dtype=bool)
    for z, model in zone_models.items():
        mask = zones == int(z)
        if not np.any(mask):
            continue
        out[mask] = model.predict_proba(x[mask])[:, 1].astype(np.float64, copy=False)
        assigned[mask] = True
    if np.any(~assigned):
        out[~assigned] = global_model.predict_proba(x[~assigned])[:, 1].astype(np.float64, copy=False)
    return out


def _subsample_train(train_df: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    if int(max_rows) <= 0 or len(train_df) <= int(max_rows):
        return train_df
    rng = np.random.default_rng(int(seed))
    pos_idx = train_df.index[train_df["flag"].astype(int) == 1].to_numpy()
    neg_idx = train_df.index[train_df["flag"].astype(int) == 0].to_numpy()
    half = int(max_rows) // 2
    n_pos = min(int(len(pos_idx)), int(half))
    n_neg = min(int(len(neg_idx)), int(max_rows - n_pos))
    if n_pos <= 0 or n_neg <= 0:
        pick = rng.choice(train_df.index.to_numpy(), size=int(max_rows), replace=False)
        return train_df.loc[pick].copy()
    pick_pos = rng.choice(pos_idx, size=int(n_pos), replace=False)
    pick_neg = rng.choice(neg_idx, size=int(n_neg), replace=False)
    pick = np.concatenate([pick_pos, pick_neg], axis=0)
    rng.shuffle(pick)
    return train_df.loc[pick].copy()


def _prepare_fold_weights(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    use_weights: bool,
    n_clusters: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not bool(use_weights):
        return train_df, val_df, test_df
    train_df, [val_df, test_df], _ = add_foldwise_kmeans_weights(
        train_df=train_df.copy(),
        other_dfs=[val_df.copy(), test_df.copy()],
        feature_columns=list(FEATURE_COLUMNS),
        n_clusters=int(n_clusters),
        seed=int(seed),
    )
    return train_df, val_df, test_df


def _event_train_val_split(event_ids: Sequence[str], val_ratio: float, seed: int) -> Tuple[List[str], List[str]]:
    event_ids = list(event_ids)
    if len(event_ids) < 3:
        return list(event_ids), []
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(np.asarray(event_ids, dtype=object))
    n_val = max(1, int(round(float(val_ratio) * len(event_ids))))
    n_val = min(n_val, len(event_ids) - 1)
    val_ids = perm[:n_val].tolist()
    train_ids = perm[n_val:].tolist()
    return train_ids, val_ids


def _prepare_land_residualized_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    args,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Sequence[str], Dict[str, object]]:
    feature_columns: Sequence[str] = list(FEATURE_COLUMNS)
    if not bool(args.land_residualize):
        return train_df, val_df, test_df, feature_columns, {"feature_mode": "raw"}

    from evaluation_protocol.land_conditioning.residualize_tb import (  # noqa: E402
        LAND_ZONES,
        apply_residuals,
        fit_background_models,
    )

    required_cols = {"window_type", "sample_date", "pixel_lat", "pixel_lon", "grid_i", "grid_j", *FEATURE_COLUMNS}
    missing_cols = [c for c in sorted(required_cols) if c not in train_df.columns]
    if missing_cols:
        raise ValueError(f"启用 --land_residualize 缺少列: {missing_cols}")

    land_train_controls = train_df[
        train_df["zone_type"].astype(int).isin(set(LAND_ZONES))
        & (train_df["flag"].astype(int) == 0)
        & (train_df["window_type"].astype(str) == "control")
    ].copy()
    if land_train_controls.empty:
        raise ValueError("land_residualize 无法拟合背景模型：训练折缺少陆地 control 样本。")

    bg_models = fit_background_models(
        land_train_controls,
        alpha=float(args.land_resid_alpha),
        include_geo_season=bool(args.land_resid_include_geo_season),
        include_era5_covariates=bool(args.land_resid_include_era5_covariates),
        zone_wise_background=bool(args.land_resid_zone_wise_background),
    )
    land_zones = set(int(z) for z in LAND_ZONES)
    resid_cols = [f"{c}_resid" for c in FEATURE_COLUMNS]

    def _apply(df_in: pd.DataFrame) -> pd.DataFrame:
        df_out = df_in.copy()
        for c in FEATURE_COLUMNS:
            df_out[f"{c}_resid"] = df_out[c].astype("float32")
        land_mask = df_out["zone_type"].astype(int).isin(land_zones).to_numpy()
        if np.any(land_mask):
            sub = apply_residuals(
                df_out.loc[land_mask].copy(),
                bg_models,
                include_geo_season=bool(args.land_resid_include_geo_season),
                include_era5_covariates=bool(args.land_resid_include_era5_covariates),
                suffix="_resid",
            )
            for c in FEATURE_COLUMNS:
                df_out.loc[land_mask, f"{c}_resid"] = sub[f"{c}_resid"].astype("float32").to_numpy()
        return df_out

    train_df = _apply(train_df)
    val_df = _apply(val_df)
    test_df = _apply(test_df)
    meta = {
        "feature_mode": "land_residualized",
        "land_zones": sorted(int(z) for z in land_zones),
        "zone_models_available": sorted(k for k in bg_models.keys() if k != "all"),
    }
    return train_df, val_df, test_df, resid_cols, meta


def _select_thresholds_and_predict(
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    bundle,
    args,
    feature_columns: Sequence[str],
) -> Tuple[np.ndarray, float, str, Dict[str, object]]:
    y_val = val_df["flag"].astype("int64").to_numpy()
    val_proba = _predict_bundle(val_df, bundle, args, feature_columns=feature_columns)
    test_proba = _predict_bundle(test_df, bundle, args, feature_columns=feature_columns)

    global_thr, global_mode = _select_threshold(
        y_val,
        val_proba,
        int(args.thr_grid_n),
        str(args.threshold_objective),
        float(args.fpr_target),
    )
    if not bool(args.zone_adaptive_thresholds):
        y_pred = (test_proba >= float(global_thr)).astype(np.int64)
        return y_pred, float(global_thr), str(global_mode), {
            "global": float(global_thr),
            "global_mode": str(global_mode),
            "by_zone": {},
        }

    zones_val = val_df["zone_type"].astype(int).to_numpy()
    zones_test = test_df["zone_type"].astype(int).to_numpy()
    by_zone: Dict[str, Dict[str, object]] = {}
    unique_zones = sorted(int(z) for z in np.unique(zones_test))
    y_pred = np.zeros(len(test_df), dtype=np.int64)

    for zone in unique_zones:
        val_mask = zones_val == int(zone)
        test_mask = zones_test == int(zone)
        use_global = False
        mode = "zone"
        zone_thr = float(global_thr)

        if int(np.sum(test_mask)) < int(args.min_zone_test_rows_for_threshold):
            use_global = True
            mode = "fallback_global_small_test"
        elif int(np.sum(val_mask)) < int(args.min_zone_val_rows_for_threshold):
            use_global = True
            mode = "fallback_global_small_val"
        else:
            y_val_z = y_val[val_mask]
            if len(np.unique(y_val_z)) < 2:
                use_global = True
                mode = "fallback_global_single_class"

        if not use_global:
            zone_thr, zone_mode = _select_threshold(
                y_val[val_mask],
                val_proba[val_mask],
                int(args.thr_grid_n),
                str(args.threshold_objective),
                float(args.fpr_target),
            )
            mode = f"zone_{zone_mode}"

        y_pred[test_mask] = (test_proba[test_mask] >= float(zone_thr)).astype(np.int64)
        by_zone[str(int(zone))] = {
            "threshold": float(zone_thr),
            "mode": str(mode),
            "n_val": int(np.sum(val_mask)),
            "n_test": int(np.sum(test_mask)),
        }

    return y_pred, float(global_thr), "zone_adaptive", {
        "global": float(global_thr),
        "global_mode": str(global_mode),
        "by_zone": by_zone,
    }


def _run_reference_event_split(
    df: pd.DataFrame,
    splits_json: Path,
    args,
) -> Dict[str, object]:
    splits = load_event_splits(splits_json)
    train_df, val_df, test_df = split_df_by_event_id(df, splits)
    train_df = _subsample_train(train_df, int(args.max_train_rows), int(args.seed))
    train_df, val_df, test_df = _prepare_fold_weights(
        train_df,
        val_df,
        test_df,
        use_weights=bool(args.use_weights),
        n_clusters=int(args.n_clusters),
        seed=int(args.seed),
    )
    train_df, val_df, test_df, feature_columns, fold_meta = _prepare_land_residualized_features(
        train_df,
        val_df,
        test_df,
        args,
    )
    bundle = _fit_model_bundle(train_df, args, seed=int(args.seed), feature_columns=feature_columns)
    test_pred, thr, thr_mode, thr_meta = _select_thresholds_and_predict(
        val_df,
        test_df,
        bundle,
        args,
        feature_columns=feature_columns,
    )
    out = _binary_metrics(test_df["flag"].astype("int64").to_numpy(), test_pred)
    out["threshold"] = float(thr)
    out["threshold_mode"] = str(thr_mode)
    out["feature_mode"] = str(fold_meta.get("feature_mode", "raw"))
    if bool(args.zone_adaptive_thresholds):
        out["zone_thresholds"] = thr_meta["by_zone"]
    out["n_events_test"] = int(len(splits.test_event_ids))
    return out


def _plot_results(df_res: pd.DataFrame, out_fig: Path, failure_mcc: float, weighted_mcc: float) -> None:
    if df_res.empty:
        return
    d = df_res.sort_values("mcc", ascending=True).copy()
    max_rows = 60
    if len(d) > max_rows:
        d = d.iloc[:max_rows].copy()
    colors = np.where(d["mcc"].to_numpy() < float(failure_mcc), "#d9534f", "#4f81bd")
    fig_h = max(5.5, 0.24 * len(d) + 1.5)
    fig, ax = plt.subplots(figsize=(10.0, fig_h), dpi=160)
    y = np.arange(len(d))
    ax.barh(y, d["mcc"].to_numpy(), color=colors, alpha=0.92)
    ax.set_yticks(y)
    ax.set_yticklabels(d["block_key"].tolist(), fontsize=8)
    ax.set_xlabel("MCC on held-out 10°×10° block")
    ax.set_title("Spatial leave-one-block-out validation (event-grouped)")
    ax.axvline(float(failure_mcc), linestyle="--", linewidth=1.2, color="#d9534f", label=f"Failure line ({failure_mcc:.2f})")
    ax.axvline(float(weighted_mcc), linestyle="--", linewidth=1.2, color="#2ca02c", label=f"Weighted mean ({weighted_mcc:.3f})")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig)
    plt.close(fig)


def _prepare_eval_df(eval_parquet: Path, aggregation: str, *, land_residualize: bool) -> pd.DataFrame:
    need_cols = ["event_id", "event_lat", "event_lon", "zone_type", "flag", *FEATURE_COLUMNS]
    if str(aggregation) == "day_mean":
        need_cols.append("sample_date")
    if bool(land_residualize):
        need_cols.extend(["window_type", "sample_date", "pixel_lat", "pixel_lon", "grid_i", "grid_j"])
    need_cols = sorted(set(need_cols))
    df = pd.read_parquet(eval_parquet, columns=need_cols)
    df = df.dropna(subset=list(FEATURE_COLUMNS)).copy()
    df["event_id"] = df["event_id"].astype(str)
    if bool(land_residualize) and str(aggregation) != "pixel":
        raise ValueError("当前 --land_residualize 仅支持 --aggregation pixel。")
    if str(aggregation) == "day_mean":
        grp_cols = ["event_id", "event_lat", "event_lon", "zone_type", "flag", "sample_date"]
        df = df.groupby(grp_cols, as_index=False)[list(FEATURE_COLUMNS)].mean()
    return df


def main():
    args = parse_args()
    eval_parquet = resolve_path(args.eval_parquet)
    splits_json = resolve_path(args.splits_json)
    era5_cache_dir = resolve_path(args.era5_daily_cache_dir)
    out_csv = resolve_path(args.out_csv)
    out_json = resolve_path(args.out_json)
    out_fig = resolve_path(args.out_fig)

    df = _prepare_eval_df(eval_parquet, str(args.aggregation), land_residualize=bool(args.land_residualize))
    if bool(args.land_residualize) and bool(args.land_resid_include_era5_covariates):
        from evaluation_protocol.land_conditioning.residualize_tb import attach_era5_covariates  # noqa: E402

        df = attach_era5_covariates(df, cache_dir=era5_cache_dir)

    ev = df[["event_id", "event_lat", "event_lon"]].drop_duplicates("event_id").copy()
    ev["block_key"] = ev.apply(
        lambda r: _block_key(float(r["event_lat"]), float(r["event_lon"]), float(args.block_size_deg)),
        axis=1,
    )
    block_to_events: Dict[str, List[str]] = ev.groupby("block_key")["event_id"].apply(list).to_dict()
    block_items = [(k, v) for k, v in block_to_events.items() if len(v) >= int(args.min_events_per_block)]
    block_items = sorted(block_items, key=lambda kv: (-len(kv[1]), kv[0]))
    if args.max_folds is not None:
        block_items = block_items[: int(args.max_folds)]
    if not block_items:
        raise ValueError("没有满足 min_events_per_block 的空间块。")

    all_events = set(ev["event_id"].tolist())
    results: List[Dict[str, object]] = []

    for idx, (block_key, test_events) in enumerate(block_items):
        test_events = sorted(set(test_events))
        train_pool = sorted(all_events - set(test_events))
        if len(train_pool) < 3:
            continue
        train_events, val_events = _event_train_val_split(
            train_pool,
            val_ratio=float(args.val_ratio),
            seed=int(args.seed) + int(idx) * 17,
        )
        if len(val_events) == 0:
            continue

        train_df = df[df["event_id"].isin(train_events)].copy()
        val_df = df[df["event_id"].isin(val_events)].copy()
        test_df = df[df["event_id"].isin(test_events)].copy()
        if (
            len(train_df) < int(args.min_train_rows_per_fold)
            or len(val_df) < int(args.min_val_rows_per_fold)
            or len(test_df) < int(args.min_test_rows_per_fold)
        ):
            continue

        train_df = _subsample_train(train_df, int(args.max_train_rows), int(args.seed) + int(idx) * 31)
        train_df, val_df, test_df = _prepare_fold_weights(
            train_df,
            val_df,
            test_df,
            use_weights=bool(args.use_weights),
            n_clusters=int(args.n_clusters),
            seed=int(args.seed) + int(idx) * 131,
        )
        train_df, val_df, test_df, feature_columns, fold_meta = _prepare_land_residualized_features(
            train_df,
            val_df,
            test_df,
            args,
        )

        bundle = _fit_model_bundle(
            train_df,
            args,
            seed=int(args.seed) + int(idx),
            feature_columns=feature_columns,
        )
        test_pred, thr, thr_mode, thr_meta = _select_thresholds_and_predict(
            val_df,
            test_df,
            bundle,
            args,
            feature_columns=feature_columns,
        )
        m = _binary_metrics(test_df["flag"].astype("int64").to_numpy(), test_pred)
        m["block_key"] = str(block_key)
        m["n_events_test"] = int(len(test_events))
        m["n_samples_test"] = int(len(test_df))
        m["threshold"] = float(thr)
        m["threshold_mode"] = str(thr_mode)
        m["feature_mode"] = str(fold_meta.get("feature_mode", "raw"))
        if bool(args.zone_adaptive_thresholds):
            m["zone_thresholds"] = json.dumps(thr_meta["by_zone"], ensure_ascii=False, sort_keys=True)
        results.append(m)

    if not results:
        raise ValueError("空间留一未产生任何有效折结果。")

    df_res = pd.DataFrame(results)
    df_res = df_res.sort_values(["mcc", "block_key"], ascending=[True, True]).reset_index(drop=True)

    weights = df_res["n_samples_test"].to_numpy(dtype=np.float64)
    weighted_mcc = float(np.average(df_res["mcc"].to_numpy(dtype=np.float64), weights=weights))
    weighted_fpr = float(np.average(df_res["fpr"].to_numpy(dtype=np.float64), weights=weights))
    median_mcc = float(np.median(df_res["mcc"].to_numpy(dtype=np.float64)))
    fail_mask = df_res["mcc"].to_numpy(dtype=np.float64) < float(args.failure_mcc)
    n_fail = int(np.sum(fail_mask))

    ref = _run_reference_event_split(df=df, splits_json=splits_json, args=args)
    ref_mcc = float(ref["mcc"])
    if abs(ref_mcc) > 1e-12:
        relative_drop_pct = float((ref_mcc - weighted_mcc) / abs(ref_mcc) * 100.0)
    else:
        relative_drop_pct = float("nan")

    summary = {
        "version": ("spatial_cv_v3" if (bool(args.land_residualize) or bool(args.zone_adaptive_thresholds)) else "spatial_cv_v2"),
        "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "config": {
            "eval_parquet": str(eval_parquet),
            "splits_json": str(splits_json),
            "aggregation": str(args.aggregation),
            "block_size_deg": float(args.block_size_deg),
            "min_events_per_block": int(args.min_events_per_block),
            "val_ratio": float(args.val_ratio),
            "max_folds": int(args.max_folds) if args.max_folds is not None else None,
            "min_train_rows_per_fold": int(args.min_train_rows_per_fold),
            "min_val_rows_per_fold": int(args.min_val_rows_per_fold),
            "min_test_rows_per_fold": int(args.min_test_rows_per_fold),
            "failure_mcc": float(args.failure_mcc),
            "max_train_rows": int(args.max_train_rows),
            "model": str(args.model),
            "zone_wise": bool(args.zone_wise),
            "min_zone_train_rows": int(args.min_zone_train_rows),
            "zone_adaptive_thresholds": bool(args.zone_adaptive_thresholds),
            "min_zone_val_rows_for_threshold": int(args.min_zone_val_rows_for_threshold),
            "min_zone_test_rows_for_threshold": int(args.min_zone_test_rows_for_threshold),
            "use_weights": bool(args.use_weights),
            "n_clusters": int(args.n_clusters),
            "thr_grid_n": int(args.thr_grid_n),
            "threshold_objective": str(args.threshold_objective),
            "fpr_target": float(args.fpr_target),
            "land_residualize": bool(args.land_residualize),
            "era5_daily_cache_dir": (str(era5_cache_dir) if bool(args.land_residualize) and bool(args.land_resid_include_era5_covariates) else None),
            "land_resid_alpha": float(args.land_resid_alpha),
            "land_resid_include_geo_season": bool(args.land_resid_include_geo_season),
            "land_resid_include_era5_covariates": bool(args.land_resid_include_era5_covariates),
            "land_resid_zone_wise_background": bool(args.land_resid_zone_wise_background),
            "seed": int(args.seed),
        },
        "n_blocks_evaluated": int(len(df_res)),
        "weighted_mean_mcc": float(weighted_mcc),
        "weighted_mean_fpr": float(weighted_fpr),
        "median_mcc": float(median_mcc),
        "n_failed_blocks_mcc_lt_threshold": int(n_fail),
        "failed_block_keys": df_res.loc[fail_mask, "block_key"].astype(str).tolist(),
        "reference_event_split": ref,
        "relative_drop_pct_vs_reference_mcc": float(relative_drop_pct),
    }

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    df_res.to_csv(out_csv, index=False)
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _plot_results(df_res, out_fig, float(args.failure_mcc), float(weighted_mcc))

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[OK] CSV: {out_csv}")
    print(f"[OK] JSON: {out_json}")
    print(f"[OK] FIG: {out_fig}")


if __name__ == "__main__":
    main()
