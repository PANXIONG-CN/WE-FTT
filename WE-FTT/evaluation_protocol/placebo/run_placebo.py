#!/usr/bin/env python3
"""
T4 Placebo 实验主入口（v1）。

目标：
- 使用事件级切分的 train 折训练一个“最小可用”分类器
- 在 test 折上得到真实 MCC
- 对指定 placebo 类型生成 MCC 零分布，并计算 p 值 / Z-score

说明（KISS）：
- 分类器默认使用 sklearn 的 SGDClassifier（logistic），以便在大样本上可运行
- 特征权重使用“训练折内拟合”的 KMeans + support_diff（见 common/weighting.py）
- Placebo 样本从原始 AMSR2 HDF5 读取（与 datasets/build_mbt_eval_dataset.py 同源）
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from dataclasses import asdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# 将 WE-FTT 根目录加入路径（保证可直接以脚本方式运行）
weftt_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if weftt_root not in sys.path:
    sys.path.insert(0, weftt_root)

from evaluation_protocol.common.amsr2 import AMSR2DailyGrid, AMSR2Spec, FEATURE_COLUMNS  # noqa: E402
from evaluation_protocol.common.catalog import EarthquakeEvent, load_events  # noqa: E402
from evaluation_protocol.common.geo import dobrovolsky_radius_km, haversine_km  # noqa: E402
from evaluation_protocol.common.jsonl import read_jsonl, write_json  # noqa: E402
from evaluation_protocol.common.logging_utils import setup_logging  # noqa: E402
from evaluation_protocol.common.metrics import Confusion, confusion_from_binary, fpr, matthews_corrcoef, wilson_ci  # noqa: E402
from evaluation_protocol.common.paths import get_repo_paths, resolve_path  # noqa: E402
from evaluation_protocol.common.sampling import (  # noqa: E402
    build_excluded_dates,
    date_window,
    sample_control_dates,
    sample_points_within_radius,
)
from evaluation_protocol.common.splits import load_event_splits, split_df_by_event_id  # noqa: E402
from evaluation_protocol.common.weighting import (  # noqa: E402
    FeatureWeightingArtifact,
    add_foldwise_kmeans_weights,
    apply_weighting_artifacts,
)


logger = logging.getLogger(__name__)

# 跨 replicate 复用：一旦某天的 AMSR2 HDF5 缺失/损坏，则后续直接跳过，避免重复 IO + 堆栈噪音
_BAD_AMSR2_DAYS: set[date] = set()


def _iter_window_days(anchor: date, pre_days: int, post_days: int):
    for d in range(-int(pre_days), int(post_days) + 1):
        yield int(d), (anchor + timedelta(days=int(d)))


def _event_index(events: Sequence[EarthquakeEvent]) -> Dict[str, EarthquakeEvent]:
    return {e.event_id: e for e in events}


def _prepare_xy(df: pd.DataFrame, *, use_weights: bool) -> Tuple[np.ndarray, np.ndarray]:
    y = df["flag"].astype("int64").to_numpy()
    x_feat = df[list(FEATURE_COLUMNS)].astype("float32").to_numpy()
    if not use_weights:
        return x_feat, y
    w_cols = [f"{c}_cluster_labels_weight" for c in FEATURE_COLUMNS]
    x_w = df[w_cols].astype("float32").to_numpy()
    return (x_feat * x_w), y


def train_classifier(train_df: pd.DataFrame, *, use_weights: bool, seed: int) -> Pipeline:
    x_train, y_train = _prepare_xy(train_df, use_weights=use_weights)
    clf = SGDClassifier(
        loss="log_loss",
        alpha=1e-4,
        max_iter=2000,
        tol=1e-3,
        class_weight="balanced",
        random_state=int(seed),
    )
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    pipe.fit(x_train, y_train)
    return pipe


def evaluate_binary(pipe: Pipeline, df: pd.DataFrame, *, use_weights: bool) -> Dict[str, float]:
    x, y_true = _prepare_xy(df, use_weights=use_weights)
    # 概率
    if hasattr(pipe, "predict_proba"):
        proba = pipe.predict_proba(x)[:, 1]
    else:
        # fallback: decision_function -> sigmoid
        s = pipe.decision_function(x)
        proba = 1.0 / (1.0 + np.exp(-s))
    y_pred = (proba >= 0.5).astype(np.int64)

    c = confusion_from_binary(y_true, y_pred)
    mcc = matthews_corrcoef(c)
    fpr_v = fpr(c)
    lo, hi = wilson_ci(successes=c.fp, n=(c.fp + c.tn))
    return {
        "mcc": float(mcc),
        "fpr": float(fpr_v),
        "fpr_ci_low": float(lo),
        "fpr_ci_high": float(hi),
        "tp": int(c.tp),
        "fp": int(c.fp),
        "tn": int(c.tn),
        "fn": int(c.fn),
    }


def _predict_proba(pipe: Pipeline, x: np.ndarray) -> np.ndarray:
    if hasattr(pipe, "predict_proba"):
        return pipe.predict_proba(x)[:, 1]
    s = pipe.decision_function(x)
    return 1.0 / (1.0 + np.exp(-s))


def _binary_metrics_at_threshold(y_true: np.ndarray, proba: np.ndarray, thr: float) -> Dict[str, float]:
    y = y_true.astype(np.int64, copy=False)
    pred_pos = (proba >= float(thr))
    y_pos = (y == 1)
    y_neg = ~y_pos

    tp = int(np.sum(pred_pos & y_pos))
    fp = int(np.sum(pred_pos & y_neg))
    tn = int(np.sum((~pred_pos) & y_neg))
    fn = int(np.sum((~pred_pos) & y_pos))
    c = Confusion(tp=tp, fp=fp, tn=tn, fn=fn)
    mcc = matthews_corrcoef(c)
    fpr_v = fpr(c)
    lo, hi = wilson_ci(successes=c.fp, n=(c.fp + c.tn))
    return {
        "thr": float(thr),
        "mcc": float(mcc),
        "fpr": float(fpr_v),
        "fpr_ci_low": float(lo),
        "fpr_ci_high": float(hi),
        "tp": int(c.tp),
        "fp": int(c.fp),
        "tn": int(c.tn),
        "fn": int(c.fn),
    }


def _select_threshold_on_val_max_mcc(*, y_true: np.ndarray, proba: np.ndarray, grid_n: int) -> Dict[str, float]:
    if y_true.ndim != 1 or proba.ndim != 1 or y_true.shape[0] != proba.shape[0]:
        raise ValueError("y_true/proba 必须为一维且长度一致。")
    grid = np.linspace(0.001, 0.999, int(max(101, grid_n))).astype(np.float32)

    best = None  # (mcc, -fpr, tp, -fp, thr)
    best_metrics = None
    for thr in grid:
        m = _binary_metrics_at_threshold(y_true, proba, float(thr))
        key = (float(m["mcc"]), -float(m["fpr"]), int(m["tp"]), -int(m["fp"]), float(m["thr"]))
        if best is None or key > best:
            best = key
            best_metrics = m
    assert best_metrics is not None
    return best_metrics


def _aggregate_day_mean_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    将像元级样本聚合为 “事件-日期” 级样本：
    group key = (event_id, sample_date, flag)，对特征取均值。

    目的：降低像元噪声与相关性，让 MCC/FPR 更接近“事件窗级”可检验信号。
    """
    need_cols = {"event_id", "sample_date", "flag", "zone_type", *FEATURE_COLUMNS}
    missing = [c for c in need_cols if c not in df.columns]
    if missing:
        raise ValueError(f"day_mean 聚合缺少列：{missing}")

    group_cols = ["event_id", "sample_date", "flag"]
    feat_df = (
        df.groupby(group_cols, as_index=False)[list(FEATURE_COLUMNS)]
        .mean()
        .astype({c: "float32" for c in FEATURE_COLUMNS})
    )
    zone_df = df.groupby(group_cols, as_index=False)["zone_type"].first()
    out = feat_df.merge(zone_df, on=group_cols, how="left")
    return out[["event_id", "sample_date", "zone_type", "flag", *FEATURE_COLUMNS]].copy()


def _sample_placebo_anchor_from_plan_row(row: dict) -> Tuple[date, float, float, int, float]:
    anchor_date = date.fromisoformat(str(row["anchor_date"]))
    lat = float(row["anchor_lat"])
    lon = float(row["anchor_lon"])
    zone_type = int(row["zone_type"])
    mag = float(row["source_mag"])
    return anchor_date, lat, lon, zone_type, mag


def _build_placebo_rows(
    plan_rows: Sequence[dict],
    *,
    amsr2_root: Path,
    pre_days: int,
    post_days: int,
    pixels_per_event_day: int,
    control_dates_per_event: int,
    doy_window: int,
    seed: int,
    lat_limit: float,
    spec: AMSR2Spec,
    excluded_dates: set[date],
    min_year: int,
    max_year: int,
) -> Iterable[dict]:
    """
    生成一个 replicate 的 placebo 样本（包含 placebo 正例窗 + 对照负例窗）。
    """
    rng = np.random.default_rng(int(seed))

    tasks_by_date: Dict[date, List[dict]] = defaultdict(list)
    for row in plan_rows:
        anchor, lat, lon, zone_type, mag = _sample_placebo_anchor_from_plan_row(row)
        radius_km = dobrovolsky_radius_km(mag)

        # placebo “正例窗”
        for day_offset, sample_day in _iter_window_days(anchor, pre_days, post_days):
            tasks_by_date[sample_day].append(
                {
                    "flag": 1,
                    "zone_type": zone_type,
                    "event_id": str(row.get("source_event_id", "")),
                    "anchor_date": anchor,
                    "anchor_lat": lat,
                    "anchor_lon": lon,
                    "radius_km": radius_km,
                    "day_offset": int(day_offset),
                    "sample_day": sample_day,
                }
            )

        # placebo “负例窗”：相对 anchor 的随机日期（同季节），同地点
        if int(control_dates_per_event) > 0:
            controls = sample_control_dates(
                event_date=anchor,
                n_controls=int(control_dates_per_event),
                rng=rng,
                min_year=int(min_year),
                max_year=int(max_year),
                doy_window=int(doy_window),
                excluded_dates=excluded_dates,
                pre_days=int(pre_days),
                post_days=int(post_days),
            )
            for c_idx, c_anchor in enumerate(controls):
                for day_offset, sample_day in _iter_window_days(c_anchor, pre_days, post_days):
                    tasks_by_date[sample_day].append(
                        {
                            "flag": 0,
                            "zone_type": zone_type,
                            "event_id": str(row.get("source_event_id", "")),
                            "anchor_date": c_anchor,
                            "anchor_lat": lat,
                            "anchor_lon": lon,
                            "radius_km": radius_km,
                            "day_offset": int(day_offset),
                            "sample_day": sample_day,
                            "control_index": int(c_idx),
                        }
                    )

    all_days = sorted(tasks_by_date.keys())
    for day in all_days:
        if day in _BAD_AMSR2_DAYS:
            continue
        tasks = tasks_by_date[day]
        try:
            with AMSR2DailyGrid(amsr2_root, day, spec=spec) as grid:
                for t in tasks:
                    try:
                        pts = sample_points_within_radius(
                            event_lat=float(t["anchor_lat"]),
                            event_lon=float(t["anchor_lon"]),
                            radius_km=float(t["radius_km"]),
                            n_points=int(pixels_per_event_day),
                            rng=rng,
                            lat_limit=float(lat_limit),
                        )
                        feats = grid.read_features_at(pts.grid_i.astype(np.int64), pts.grid_j.astype(np.int64))
                        n = int(pts.grid_i.size)
                        for k in range(n):
                            row = {
                                "event_id": str(t.get("event_id", "")),
                                "flag": int(t["flag"]),
                                "zone_type": int(t["zone_type"]),
                                "anchor_date": str(t["anchor_date"]),
                                "sample_date": day.isoformat(),
                                "day_offset": int(t["day_offset"]),
                                "grid_i": int(pts.grid_i[k]),
                                "grid_j": int(pts.grid_j[k]),
                                "pixel_lat": float(pts.pixel_lat[k]),
                                "pixel_lon": float(pts.pixel_lon[k]),
                            }
                            for col in FEATURE_COLUMNS:
                                row[col] = float(feats[col][k]) if feats[col].size > 0 else float("nan")
                            yield row
                    except Exception:
                        logger.exception(
                            "placebo 任务失败，跳过：day=%s placebo_flag=%s zone_type=%s",
                            day.isoformat(),
                            str(t.get("flag", "")),
                            str(t.get("zone_type", "")),
                        )
                        continue
        except FileNotFoundError:
            _BAD_AMSR2_DAYS.add(day)
            continue
        except Exception as e:
            _BAD_AMSR2_DAYS.add(day)
            # 异常信息中应包含具体文件路径（AMSR2DailyGrid.open 已包装）
            logger.warning("placebo 日期失败，跳过：day=%s err=%s", day.isoformat(), e)
            continue


def _apply_weighting_artifacts_to_matrix(
    x: np.ndarray,
    artifacts: Sequence[FeatureWeightingArtifact],
    *,
    feature_columns: Sequence[str],
) -> np.ndarray:
    """
    将 artifacts 中的“簇权重”直接作用到特征矩阵（X = TB * weight）。

    说明：
    - 复用 apply_weighting_artifacts 的核心规则：1D 最近中心 -> cluster weight
    - 避免为每个 replicate 构造 DataFrame，显著降低 placebo 开销
    """
    if x.ndim != 2 or x.shape[1] != len(feature_columns):
        raise ValueError("x shape 不匹配 feature_columns。")
    if not artifacts:
        return x

    out = x.astype(np.float32, copy=True)
    by_feature = {a.feature: a for a in artifacts}
    for feat_idx, feat in enumerate(feature_columns):
        a = by_feature.get(str(feat), None)
        if a is None:
            raise ValueError(f"未找到 feature={feat} 的 weighting artifact。")
        centers = np.asarray(a.cluster_centers, dtype=np.float32).reshape(1, -1)  # (1,k)
        weights = np.asarray(a.cluster_weights, dtype=np.float32).reshape(-1)  # (k,)
        col = out[:, feat_idx : feat_idx + 1]  # (n,1)
        idx = np.argmin(np.abs(col - centers), axis=1)
        out[:, feat_idx] *= weights[idx]
    return out


def _evaluate_binary_xy(pipe: Pipeline, x: np.ndarray, y_true: np.ndarray, *, thr: float = 0.5) -> Dict[str, float]:
    if x.ndim != 2:
        raise ValueError("x 必须为二维数组。")
    if y_true.ndim != 1 or y_true.shape[0] != x.shape[0]:
        raise ValueError("y_true 必须为一维且长度与 x 匹配。")
    if x.shape[0] == 0:
        return {"mcc": 0.0, "fpr": 0.0, "fpr_ci_low": 0.0, "fpr_ci_high": 0.0, "tp": 0, "fp": 0, "tn": 0, "fn": 0}

    proba = _predict_proba(pipe, x).astype(np.float64, copy=False)
    m = _binary_metrics_at_threshold(y_true, proba, float(thr))
    m.pop("thr", None)
    return m


def _build_placebo_xy(
    plan_rows: Sequence[dict],
    *,
    amsr2_root: Path,
    pre_days: int,
    post_days: int,
    pixels_per_event_day: int,
    control_dates_per_event: int,
    doy_window: int,
    seed: int,
    lat_limit: float,
    spec: AMSR2Spec,
    excluded_dates: set[date],
    min_year: int,
    max_year: int,
    aggregation: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成一个 replicate 的 (X, y)：
    - X: (n, n_features) float32
    - y: (n,) int64（flag=0/1）

    相比 _build_placebo_rows：
    - 不构造逐像元 dict，不构造 DataFrame
    - 在读取当天 HDF5 后，直接拼接 numpy 数组（显著提速）
    """
    aggregation = str(aggregation).strip().lower()
    if aggregation not in {"pixel", "day_mean"}:
        raise ValueError(f"不支持的 aggregation: {aggregation}（仅支持 pixel/day_mean）")

    rng = np.random.default_rng(int(seed))

    tasks_by_date: Dict[date, List[dict]] = defaultdict(list)
    for row in plan_rows:
        anchor, lat, lon, zone_type, mag = _sample_placebo_anchor_from_plan_row(row)
        event_id = str(row.get("source_event_id", ""))
        radius_km = dobrovolsky_radius_km(mag)

        for day_offset, sample_day in _iter_window_days(anchor, pre_days, post_days):
            tasks_by_date[sample_day].append(
                {
                    "flag": 1,
                    "zone_type": zone_type,
                    "event_id": event_id,
                    "anchor_date": anchor,
                    "anchor_lat": lat,
                    "anchor_lon": lon,
                    "radius_km": radius_km,
                    "day_offset": int(day_offset),
                }
            )

        if int(control_dates_per_event) > 0:
            controls = sample_control_dates(
                event_date=anchor,
                n_controls=int(control_dates_per_event),
                rng=rng,
                min_year=int(min_year),
                max_year=int(max_year),
                doy_window=int(doy_window),
                excluded_dates=excluded_dates,
                pre_days=int(pre_days),
                post_days=int(post_days),
            )
            for c_anchor in controls:
                for day_offset, sample_day in _iter_window_days(c_anchor, pre_days, post_days):
                    tasks_by_date[sample_day].append(
                        {
                            "flag": 0,
                            "zone_type": zone_type,
                            "event_id": event_id,
                            "anchor_date": c_anchor,
                            "anchor_lat": lat,
                            "anchor_lon": lon,
                            "radius_km": radius_km,
                            "day_offset": int(day_offset),
                        }
                    )

    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    sum_by_key: Dict[Tuple[str, str, int], np.ndarray] = {}
    cnt_by_key: Dict[Tuple[str, str, int], int] = {}

    all_days = sorted(tasks_by_date.keys())
    for day in all_days:
        if day in _BAD_AMSR2_DAYS:
            continue
        tasks = tasks_by_date[day]
        try:
            with AMSR2DailyGrid(amsr2_root, day, spec=spec) as grid:
                for t in tasks:
                    try:
                        pts = sample_points_within_radius(
                            event_lat=float(t["anchor_lat"]),
                            event_lon=float(t["anchor_lon"]),
                            radius_km=float(t["radius_km"]),
                            n_points=int(pixels_per_event_day),
                            rng=rng,
                            lat_limit=float(lat_limit),
                        )
                        feats = grid.read_features_at(pts.grid_i.astype(np.int64), pts.grid_j.astype(np.int64))
                        x_task = np.stack([feats[c].astype(np.float32, copy=False) for c in FEATURE_COLUMNS], axis=1)
                        mask = np.isfinite(x_task).all(axis=1)
                        if not np.any(mask):
                            continue
                        x_keep = x_task[mask]
                        flag = int(t["flag"])
                        if aggregation == "day_mean":
                            k = (str(t.get("event_id", "")), day.isoformat(), int(flag))
                            acc = sum_by_key.get(k, None)
                            if acc is None:
                                acc = np.zeros((len(FEATURE_COLUMNS),), dtype=np.float64)
                                sum_by_key[k] = acc
                            acc += np.sum(x_keep, axis=0, dtype=np.float64)
                            cnt_by_key[k] = int(cnt_by_key.get(k, 0)) + int(x_keep.shape[0])
                        else:
                            y_keep = np.full((x_keep.shape[0],), int(flag), dtype=np.int64)
                            xs.append(x_keep)
                            ys.append(y_keep)
                    except Exception:
                        logger.exception(
                            "placebo 任务失败，跳过：day=%s placebo_flag=%s zone_type=%s",
                            day.isoformat(),
                            str(t.get("flag", "")),
                            str(t.get("zone_type", "")),
                        )
                        continue
        except FileNotFoundError:
            _BAD_AMSR2_DAYS.add(day)
            continue
        except Exception as e:
            _BAD_AMSR2_DAYS.add(day)
            logger.warning("placebo 日期失败，跳过：day=%s err=%s", day.isoformat(), e)
            continue

    if aggregation == "day_mean":
        if not sum_by_key:
            return (np.zeros((0, len(FEATURE_COLUMNS)), dtype=np.float32), np.zeros((0,), dtype=np.int64))
        keys = sorted(sum_by_key.keys())
        x = np.stack([sum_by_key[k] / float(cnt_by_key[k]) for k in keys], axis=0).astype(np.float32, copy=False)
        y = np.asarray([int(k[2]) for k in keys], dtype=np.int64)
        return x, y
    else:
        if not xs:
            return (np.zeros((0, len(FEATURE_COLUMNS)), dtype=np.float32), np.zeros((0,), dtype=np.int64))
        x = np.concatenate(xs, axis=0).astype(np.float32, copy=False)
        y = np.concatenate(ys, axis=0).astype(np.int64, copy=False)
        return x, y


def _build_placebo_xy_from_eval_dates(
    plan_rows: Sequence[dict],
    *,
    amsr2_root: Path,
    pixels_per_event_day: int,
    seed: int,
    lat_limit: float,
    spec: AMSR2Spec,
    dates_by_event: Dict[str, Sequence[Tuple[date, int]]],
    aggregation: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    使用 eval_parquet（test 折）中已有的 (event_id, sample_date, flag) 作为“日期/标签清单”，
    仅对 location 做随机化（random_location），生成一个 replicate 的 (X, y)。

    目的：让 random_location placebo 与真实评估使用同一套日期/标签分布，避免因“重新采样 control dates”
    引入额外的跨年/缺测偏差，从而更干净地检验“位置随机化后是否仍有区分能力”。
    """
    aggregation = str(aggregation).strip().lower()
    if aggregation != "day_mean":
        raise ValueError("eval_dates 模式仅支持 aggregation=day_mean（保证日期/标签为事件-日期级）。")

    rng = np.random.default_rng(int(seed))

    tasks_by_date: Dict[date, List[dict]] = defaultdict(list)
    skipped_events = 0
    for row in plan_rows:
        _anchor_date, lat, lon, zone_type, mag = _sample_placebo_anchor_from_plan_row(row)
        event_id = str(row.get("source_event_id", ""))
        date_flags = dates_by_event.get(event_id, None)
        if not date_flags:
            skipped_events += 1
            continue
        radius_km = dobrovolsky_radius_km(mag)
        for sample_day, flag in date_flags:
            tasks_by_date[sample_day].append(
                {
                    "flag": int(flag),
                    "zone_type": int(zone_type),
                    "event_id": event_id,
                    "anchor_lat": float(lat),
                    "anchor_lon": float(lon),
                    "radius_km": float(radius_km),
                }
            )
    if skipped_events:
        logger.info("eval_dates: 跳过无日期清单的事件数=%d（可能由 zone_filter 导致）", int(skipped_events))

    sum_by_key: Dict[Tuple[str, str, int], np.ndarray] = {}
    cnt_by_key: Dict[Tuple[str, str, int], int] = {}

    all_days = sorted(tasks_by_date.keys())
    for day in all_days:
        if day in _BAD_AMSR2_DAYS:
            continue
        tasks = tasks_by_date[day]
        try:
            with AMSR2DailyGrid(amsr2_root, day, spec=spec) as grid:
                for t in tasks:
                    try:
                        pts = sample_points_within_radius(
                            event_lat=float(t["anchor_lat"]),
                            event_lon=float(t["anchor_lon"]),
                            radius_km=float(t["radius_km"]),
                            n_points=int(pixels_per_event_day),
                            rng=rng,
                            lat_limit=float(lat_limit),
                        )
                        feats = grid.read_features_at(pts.grid_i.astype(np.int64), pts.grid_j.astype(np.int64))
                        x_task = np.stack([feats[c].astype(np.float32, copy=False) for c in FEATURE_COLUMNS], axis=1)
                        mask = np.isfinite(x_task).all(axis=1)
                        if not np.any(mask):
                            continue
                        x_keep = x_task[mask]
                        flag = int(t["flag"])
                        k = (str(t.get("event_id", "")), day.isoformat(), int(flag))
                        acc = sum_by_key.get(k, None)
                        if acc is None:
                            acc = np.zeros((len(FEATURE_COLUMNS),), dtype=np.float64)
                            sum_by_key[k] = acc
                        acc += np.sum(x_keep, axis=0, dtype=np.float64)
                        cnt_by_key[k] = int(cnt_by_key.get(k, 0)) + int(x_keep.shape[0])
                    except Exception:
                        logger.exception(
                            "placebo 任务失败（eval_dates），跳过：day=%s placebo_flag=%s zone_type=%s",
                            day.isoformat(),
                            str(t.get("flag", "")),
                            str(t.get("zone_type", "")),
                        )
                        continue
        except FileNotFoundError:
            _BAD_AMSR2_DAYS.add(day)
            continue
        except Exception as e:
            _BAD_AMSR2_DAYS.add(day)
            logger.warning("placebo 日期失败（eval_dates），跳过：day=%s err=%s", day.isoformat(), e)
            continue

    if not sum_by_key:
        return (np.zeros((0, len(FEATURE_COLUMNS)), dtype=np.float32), np.zeros((0,), dtype=np.int64))
    keys = sorted(sum_by_key.keys())
    x = np.stack([sum_by_key[k] / float(cnt_by_key[k]) for k in keys], axis=0).astype(np.float32, copy=False)
    y = np.asarray([int(k[2]) for k in keys], dtype=np.int64)
    return x, y


def parse_args():
    repo = get_repo_paths()
    default_catalog = repo.weftt_root / "data" / "raw" / "earthquake_catalog.csv"
    default_splits = repo.eval_root / "data_splits" / "event_grouped_splits_v1.json"
    default_eval = repo.eval_root / "datasets" / "mbt_eval_samples_v1.parquet"
    default_out = repo.eval_root / "placebo" / "results" / "placebo_results_v1.json"

    p = argparse.ArgumentParser(description="Run placebo experiments (v1)")
    p.add_argument("--catalog_csv", type=str, default=str(default_catalog))
    p.add_argument("--splits_json", type=str, default=str(default_splits))
    p.add_argument("--eval_parquet", type=str, default=str(default_eval), help="真实评估数据集（需包含 flag=0/1）")
    p.add_argument("--amsr2_root", type=str, required=True)

    p.add_argument("--placebo_plan_jsonl", type=str, required=True, help="generate_placebos.py 产物（jsonl）")
    p.add_argument(
        "--placebo_type",
        type=str,
        required=True,
        choices=["random_date", "time_shift", "random_location", "nonseismic_extreme"],
    )
    p.add_argument("--n_repeats", type=int, default=100)

    p.add_argument("--pre_days", type=int, default=20)
    p.add_argument("--post_days", type=int, default=10)
    p.add_argument("--pixels_per_event_day", type=int, default=200)
    p.add_argument("--control_dates_per_event", type=int, default=1, help="placebo 数据集中每个事件的负例窗数量")
    p.add_argument("--doy_window", type=int, default=15)
    p.add_argument(
        "--random_location_date_mode",
        type=str,
        default="resample_controls",
        choices=["resample_controls", "eval_dates"],
        help="random_location 的日期/标签来源：resample_controls=重采样 control dates；eval_dates=复用 test 折已有 (sample_date,flag)，仅随机化位置",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lat_limit", type=float, default=70.0)
    p.add_argument(
        "--zone_filter",
        type=str,
        default="all",
        choices=["all", "land", "ocean"],
        help="训练/评估使用的 zone_type 子集：all=0..4；land=1..4；ocean=0",
    )
    p.add_argument(
        "--aggregation",
        type=str,
        default="pixel",
        choices=["pixel", "day_mean"],
        help="样本粒度：pixel=像元级；day_mean=按(event_id,sample_date,flag)聚合为日均样本",
    )
    p.add_argument(
        "--min_rep_n",
        type=int,
        default=0,
        help="placebo replicate 的最小有效样本数（pixel=像元数；day_mean=日均样本数），低于则跳过且不纳入置换分布",
    )
    p.add_argument("--n_clusters", type=int, default=5, help="训练折内离散化簇数（用于权重映射）")
    p.add_argument("--use_weights", action="store_true", help="启用权重增强（X = TB * weight）")

    p.add_argument(
        "--thr_select_mode",
        type=str,
        default="val_mcc",
        choices=["val_mcc", "fixed"],
        help="分类阈值选择策略：val_mcc=在 val 上选择使 MCC 最大的阈值；fixed=固定阈值 --thr_fixed",
    )
    p.add_argument("--thr_fixed", type=float, default=0.5, help="thr_select_mode=fixed 时使用的固定阈值")
    p.add_argument("--thr_grid_n", type=int, default=999, help="thr_select_mode=val_mcc 时的阈值网格数量")
    p.add_argument("--tag", type=str, default="v1", help="用于表/图文件名后缀（避免覆盖旧产物）")

    p.add_argument("--out_json", type=str, default=str(default_out))
    p.add_argument("--log_file", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging(log_file=args.log_file)

    catalog_csv = resolve_path(args.catalog_csv)
    splits_json = resolve_path(args.splits_json)
    eval_parquet = resolve_path(args.eval_parquet)
    placebo_plan_jsonl = resolve_path(args.placebo_plan_jsonl)
    amsr2_root = Path(args.amsr2_root)
    out_json = resolve_path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    repo = get_repo_paths()
    tables_dir = repo.eval_root / "placebo" / "tables"
    figures_dir = repo.eval_root / "placebo" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    events = load_events(catalog_csv, min_mag=7.0, require_zone=True)
    splits = load_event_splits(splits_json)

    # 真实评估数据：必须含 flag=0/1
    need_cols = ["event_id", "flag", "zone_type", *FEATURE_COLUMNS]
    if str(args.aggregation).strip().lower() == "day_mean":
        need_cols = ["event_id", "sample_date", "flag", "zone_type", *FEATURE_COLUMNS]
    df = pd.read_parquet(eval_parquet, columns=need_cols)
    zone_filter = str(args.zone_filter).strip().lower()
    if zone_filter == "land":
        df = df[df["zone_type"].astype(int).isin([1, 2, 3, 4])].copy()
    elif zone_filter == "ocean":
        df = df[df["zone_type"].astype(int) == 0].copy()
    if df.empty:
        raise ValueError(f"zone_filter={zone_filter} 下样本为空，无法训练/评估。")

    aggregation = str(args.aggregation).strip().lower()
    if aggregation == "day_mean":
        df = _aggregate_day_mean_df(df)
        if df.empty:
            raise ValueError("day_mean 聚合后样本为空，无法训练/评估。")

    if df["flag"].nunique() < 2:
        raise ValueError("eval_parquet 仅包含单一 flag，无法训练/评估。请在构建数据集时加入 control_dates_per_event。")

    logger.info("data prepared: zone_filter=%s aggregation=%s n=%d", str(zone_filter), str(aggregation), int(len(df)))

    train_df, val_df, test_df = split_df_by_event_id(df, splits)
    test_event_ids_present = set(test_df["event_id"].astype(str).unique().tolist())

    # 训练折内权重映射（可选）
    artifacts: List[FeatureWeightingArtifact] = []
    if bool(args.use_weights):
        train_df, (val_df, test_df), artifacts = add_foldwise_kmeans_weights(
            train_df,
            [val_df, test_df],
            feature_columns=FEATURE_COLUMNS,
            n_clusters=int(args.n_clusters),
            seed=int(args.seed),
        )

    # 训练模型
    pipe = train_classifier(train_df, use_weights=bool(args.use_weights), seed=int(args.seed))

    # 阈值选择：默认在 val 上选阈值（max MCC），避免固定 0.5 导致 MCC/FPR 失真
    x_val, y_val = _prepare_xy(val_df, use_weights=bool(args.use_weights))
    p_val = _predict_proba(pipe, x_val).astype(np.float64, copy=False)

    thr_mode = str(args.thr_select_mode).strip().lower()
    if thr_mode == "fixed":
        thr_selected = float(args.thr_fixed)
        val_at_thr = _binary_metrics_at_threshold(y_val, p_val, thr_selected)
    else:
        val_at_thr = _select_threshold_on_val_max_mcc(y_true=y_val, proba=p_val, grid_n=int(args.thr_grid_n))
        thr_selected = float(val_at_thr["thr"])

    logger.info(
        "阈值选择：mode=%s thr=%.4f | val MCC=%.4f FPR=%.4f (tp=%d fp=%d tn=%d fn=%d)",
        thr_mode,
        thr_selected,
        float(val_at_thr["mcc"]),
        float(val_at_thr["fpr"]),
        int(val_at_thr["tp"]),
        int(val_at_thr["fp"]),
        int(val_at_thr["tn"]),
        int(val_at_thr["fn"]),
    )

    x_test, y_test = _prepare_xy(test_df, use_weights=bool(args.use_weights))
    real_metrics = _evaluate_binary_xy(pipe, x_test, y_true=y_test, thr=thr_selected)
    logger.info("真实 test MCC=%.4f FPR=%.4f", real_metrics["mcc"], real_metrics["fpr"])

    # 读取 placebo plan，并按 replicate 组织
    plan_all = [
        r
        for r in read_jsonl(placebo_plan_jsonl)
        if str(r.get("placebo_type")) == str(args.placebo_type)
        and str(r.get("source_event_id")) in test_event_ids_present
    ]
    by_rep: Dict[int, List[dict]] = defaultdict(list)
    for r in plan_all:
        by_rep[int(r["replicate_id"])].append(r)
    reps = sorted(by_rep.keys())[: int(args.n_repeats)]
    if not reps:
        raise ValueError("placebo_plan_jsonl 中未找到匹配的 placebo_type/replicate。")

    random_location_date_mode = str(args.random_location_date_mode).strip().lower()
    dates_by_event: Dict[str, List[Tuple[date, int]]] = {}
    if random_location_date_mode != "resample_controls" and str(args.placebo_type) != "random_location":
        raise ValueError("--random_location_date_mode 仅在 placebo_type=random_location 时可用。")
    if str(args.placebo_type) == "random_location" and random_location_date_mode == "eval_dates":
        if aggregation != "day_mean":
            raise ValueError("random_location_date_mode=eval_dates 需要 aggregation=day_mean。")
        tmp = test_df[["event_id", "sample_date", "flag"]].copy()
        tmp["event_id"] = tmp["event_id"].astype(str)
        tmp["flag"] = tmp["flag"].astype(int)
        tmp["sample_date"] = pd.to_datetime(tmp["sample_date"]).dt.date
        for eid, g in tmp.groupby("event_id"):
            pairs = sorted({(d, int(f)) for d, f in zip(g["sample_date"], g["flag"])}, key=lambda x: (x[0], x[1]))
            dates_by_event[str(eid)] = list(pairs)
        logger.info(
            "random_location_date_mode=eval_dates: test 折日期/标签对数=%d（events=%d）",
            int(sum(len(v) for v in dates_by_event.values())),
            int(len(dates_by_event)),
        )

    # 为 placebo 负例采样准备 excluded_dates（仅用于控制日期）
    excluded_dates = build_excluded_dates([(e.event_date, int(args.pre_days), int(args.post_days)) for e in events])
    years = [e.event_date.year for e in events]
    min_year, max_year = (min(years), max(years)) if years else (2013, 2023)

    spec = AMSR2Spec(orbit_tag="EQMD", mean_tag="01D")

    placebo_mcc: List[float] = []
    placebo_rep_ids: List[int] = []
    placebo_rep_n: List[int] = []
    for rep in reps:
        if str(args.placebo_type) == "random_location" and random_location_date_mode == "eval_dates":
            x, y = _build_placebo_xy_from_eval_dates(
                by_rep[rep],
                amsr2_root=amsr2_root,
                pixels_per_event_day=int(args.pixels_per_event_day),
                seed=int(args.seed) + int(rep) + 7,
                lat_limit=float(args.lat_limit),
                spec=spec,
                dates_by_event=dates_by_event,
                aggregation=aggregation,
            )
        else:
            x, y = _build_placebo_xy(
                by_rep[rep],
                amsr2_root=amsr2_root,
                pre_days=int(args.pre_days),
                post_days=int(args.post_days),
                pixels_per_event_day=int(args.pixels_per_event_day),
                control_dates_per_event=int(args.control_dates_per_event),
                doy_window=int(args.doy_window),
                seed=int(args.seed) + int(rep) + 7,
                lat_limit=float(args.lat_limit),
                spec=spec,
                excluded_dates=excluded_dates,
                min_year=int(min_year),
                max_year=int(max_year),
                aggregation=aggregation,
            )
        rep_n = int(y.shape[0])
        if int(args.min_rep_n) > 0 and rep_n < int(args.min_rep_n):
            logger.warning(
                "placebo rep=%d 样本量不足，跳过：n=%d < min_rep_n=%d",
                int(rep),
                int(rep_n),
                int(args.min_rep_n),
            )
            continue
        if bool(args.use_weights):
            x = _apply_weighting_artifacts_to_matrix(x, artifacts, feature_columns=FEATURE_COLUMNS)
        m = _evaluate_binary_xy(pipe, x, y_true=y, thr=thr_selected)
        placebo_mcc.append(float(m["mcc"]))
        placebo_rep_ids.append(int(rep))
        placebo_rep_n.append(int(rep_n))
        logger.info("placebo rep=%d n=%d MCC=%.4f", int(rep), int(rep_n), m["mcc"])

    placebo_arr = np.asarray(placebo_mcc, dtype=np.float64)
    mu = float(np.mean(placebo_arr)) if placebo_arr.size else 0.0
    sd = float(np.std(placebo_arr, ddof=1)) if placebo_arr.size >= 2 else 0.0
    z = (real_metrics["mcc"] - mu) / (sd + 1e-12)
    p = (1.0 + float(np.sum(placebo_arr >= float(real_metrics["mcc"])))) / (float(placebo_arr.size) + 1.0)

    # 结果表/图（单 placebo_type）
    dist_csv = out_json.with_suffix(".mcc.csv")
    pd.DataFrame({"replicate_id": placebo_rep_ids, "n_samples": placebo_rep_n, "mcc": placebo_mcc}).to_csv(dist_csv, index=False)

    tag = str(args.tag).strip() or "v1"
    tab_csv = tables_dir / f"tab_s6_placebo_{args.placebo_type}_{tag}.csv"
    pd.DataFrame(
        [
            {
                "placebo_type": str(args.placebo_type),
                "n_repeats": int(len(placebo_mcc)),
                "real_mcc": float(real_metrics["mcc"]),
                "placebo_mean_mcc": float(mu),
                "placebo_std_mcc": float(sd),
                "z_score": float(z),
                "p_value_ge": float(p),
                "zone_filter": str(zone_filter),
                "aggregation": str(aggregation),
                "thr_select_mode": str(thr_mode),
                "thr_selected_on_val": float(thr_selected),
            }
        ]
    ).to_csv(tab_csv, index=False)

    fig_png = figures_dir / f"fig_s5_mcc_{args.placebo_type}_{tag}.png"
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6.5, 3.2))
        plt.boxplot(placebo_mcc, vert=False, showfliers=False)
        plt.scatter(placebo_mcc, np.ones(len(placebo_mcc)), s=10, alpha=0.6)
        plt.axvline(float(real_metrics["mcc"]), color="red", linewidth=2, label="Real")
        plt.yticks([1], [str(args.placebo_type)])
        plt.xlabel("MCC")
        plt.title("Real vs Placebo MCC")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(fig_png, dpi=200)
        plt.close()
    except Exception as e:
        logger.warning("绘图失败（将只保留表格/JSON）：%s", e)

    payload = {
        "version": f"placebo_results_{tag}",
        "generated_at_utc": datetime.utcnow().isoformat() + "Z",
        "inputs": {
            "eval_parquet": str(eval_parquet),
            "splits_json": str(splits_json),
            "placebo_plan_jsonl": str(placebo_plan_jsonl),
            "amsr2_root": str(amsr2_root),
        },
        "params": {
            "placebo_type": str(args.placebo_type),
            "n_repeats": int(len(placebo_mcc)),
            "pre_days": int(args.pre_days),
            "post_days": int(args.post_days),
            "pixels_per_event_day": int(args.pixels_per_event_day),
            "control_dates_per_event": int(args.control_dates_per_event),
            "doy_window": int(args.doy_window),
            "seed": int(args.seed),
            "lat_limit": float(args.lat_limit),
            "zone_filter": str(zone_filter),
            "aggregation": str(aggregation),
            "use_weights": bool(args.use_weights),
            "n_clusters": int(args.n_clusters),
            "min_rep_n": int(args.min_rep_n),
            "thr_select_mode": str(thr_mode),
            "thr_fixed": float(args.thr_fixed),
            "thr_grid_n": int(args.thr_grid_n),
            "tag": str(tag),
        },
        "selection": {
            "thr_selected_on_val": float(thr_selected),
            "val_at_thr": val_at_thr,
        },
        "real_test": real_metrics,
        "placebo": {
            "replicate_id": [int(x) for x in placebo_rep_ids],
            "n_samples": [int(x) for x in placebo_rep_n],
            "mcc_values": [float(x) for x in placebo_mcc],
            "mean": mu,
            "std": sd,
            "z_score": float(z),
            "p_value_ge": float(p),
        },
        "artifacts": {
            "mcc_csv": str(dist_csv),
            "table_csv": str(tab_csv),
            "figure_png": str(fig_png),
        },
        "weighting_artifacts": [asdict(a) for a in artifacts] if artifacts else [],
    }
    write_json(out_json, payload)


if __name__ == "__main__":
    main()
