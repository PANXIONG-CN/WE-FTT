#!/usr/bin/env python3
"""
T6 海洋区（Zone A / type=0）ERA5 条件化：用 WE-FTT（轻量训练）做可复现评估（v1）。

为什么需要这个脚本？
- 现有 ocean_conditioning.py 使用的是最小可用的 sklearn 线性分类器，无法代表 WE-FTT。
- 该脚本在 evaluation_protocol 的事件级切分框架下，训练一个可控规模的 WE-FTT（二分类：flag）
  并评估 ERA5 硬掩膜（风速/降水阈值）前后 Zone A 的 FPR/MCC（含 Wilson CI）。

注意：
- 本脚本只用于评估协议/补充实验，不替代主训练管线。
- 默认采用较小模型（可通过参数调节），以便在 CPU 环境可运行。
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

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
from evaluation_protocol.common.weighting import add_foldwise_kmeans_weights, artifacts_to_jsonable, apply_weighting_artifacts  # noqa: E402
from evaluation_protocol.era5.ocean_residualize import attach_era5_covariates  # noqa: E402


logger = logging.getLogger(__name__)


def _require_torch():
    try:
        import torch  # noqa: F401
    except Exception as e:
        raise ImportError(
            "缺少依赖 torch。请在 WE-FTT/.venv 中安装 torch 后重试。"
        ) from e


@dataclass(frozen=True)
class Metrics:
    mcc: float
    fpr: float
    fpr_ci_low: float
    fpr_ci_high: float
    tp: int
    fp: int
    tn: int
    fn: int
    n: int


def _confusion_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Metrics:
    c = confusion_from_binary(y_true, y_pred)
    lo, hi = wilson_ci(successes=c.fp, n=(c.fp + c.tn))
    return Metrics(
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


def _standardize_fit(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = np.mean(x, axis=0, dtype=np.float64).astype(np.float32)
    std = np.std(x, axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(std, 1e-6).astype(np.float32)
    return mean, std


def _standardize_apply(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x - mean) / std).astype(np.float32)


def _build_arrays(
    df: pd.DataFrame,
    *,
    use_weights: bool,
    mean: Optional[np.ndarray],
    std: Optional[np.ndarray],
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    x = df[list(FEATURE_COLUMNS)].astype("float32").to_numpy(copy=True)
    if mean is not None and std is not None:
        x = _standardize_apply(x, mean, std)

    w = None
    if bool(use_weights):
        w_cols = [f"{c}_cluster_labels_weight" for c in FEATURE_COLUMNS]
        w = df[w_cols].astype("float32").to_numpy(copy=True)

    y = df["flag"].astype("int64").to_numpy(copy=True)
    return x, w, y


def _predict_proba_pos(model, x: np.ndarray, w: Optional[np.ndarray], *, batch_size: int) -> np.ndarray:
    import torch

    model.eval()
    out: List[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, x.shape[0], int(batch_size)):
            xb = torch.from_numpy(x[i : i + batch_size]).float()
            if w is not None:
                wb = torch.from_numpy(w[i : i + batch_size]).float()
                logits = model(xb, wb)
            else:
                logits = model(xb)
            prob = torch.softmax(logits, dim=1)[:, 1]
            out.append(prob.cpu().numpy())
    return np.concatenate(out, axis=0)


def _pick_threshold_for_fpr(
    y_true: np.ndarray,
    proba: np.ndarray,
    *,
    fpr_max: float,
    grid: np.ndarray,
) -> Optional[float]:
    best_thr = None
    best_mcc = -1e9
    for thr in grid:
        y_pred = (proba >= float(thr)).astype(np.int64)
        m = _confusion_metrics(y_true, y_pred)
        if m.fpr <= float(fpr_max) and m.mcc > best_mcc:
            best_mcc = float(m.mcc)
            best_thr = float(thr)
    return best_thr


def apply_era5_mask(
    df: pd.DataFrame,
    *,
    wind_thresh: float,
    precip_thresh: float,
) -> pd.DataFrame:
    """
    样本级硬掩膜：按行保留 ERA5 风速/降水均不超过阈值的样本（NaN 视为保留）。
    ERA5 值由 attach_era5_covariates 写入列：
    - era5_wind_speed（m/s）
    - era5_precip_mm_day（mm/day）
    """
    ws = df["era5_wind_speed"].astype("float32")
    pr = df["era5_precip_mm_day"].astype("float32")
    keep = ((ws <= float(wind_thresh)) | ws.isna()) & ((pr <= float(precip_thresh)) | pr.isna())
    return df[keep].copy()


def parse_args():
    repo = get_repo_paths()
    default_eval = repo.eval_root / "datasets" / "mbt_eval_samples_v1.parquet"
    default_splits = repo.eval_root / "data_splits" / "event_grouped_splits_v1.json"
    default_cache = repo.eval_root / "era5" / "cache" / "daily"

    default_out = repo.eval_root / "era5" / "results" / "ocean_conditioning_weftt_v1.json"
    default_tab = repo.eval_root / "era5" / "tables" / "tab_s7_era5_ocean_mask_weftt_v1.csv"
    default_fig = repo.eval_root / "era5" / "figures" / "fig_s6_era5_ocean_fpr_weftt_v1.png"
    default_ckpt = repo.eval_root / "era5" / "results" / "ocean_conditioning_weftt_v1.ckpt.pth"

    p = argparse.ArgumentParser(description="ERA5 ocean conditioning with WE-FTT (train+eval, v1)")
    p.add_argument("--eval_parquet", type=str, default=str(default_eval))
    p.add_argument("--splits_json", type=str, default=str(default_splits))
    p.add_argument("--era5_daily_cache_dir", type=str, default=str(default_cache))

    p.add_argument("--wind_thresh", type=float, default=10.0, help="m/s（硬掩膜阈值）")
    p.add_argument("--precip_thresh", type=float, default=5.0, help="mm/day（硬掩膜阈值）")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use_weights", action="store_true", help="使用 fold-wise KMeans 权重作为 WE-FTT weights 输入")
    p.add_argument("--n_clusters", type=int, default=5)

    # 训练参数（CPU 可跑的默认值）
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--early_stop_patience", type=int, default=2)

    # 轻量模型结构（可按需调大）
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--ffn_hidden_dim", type=int, default=256)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--dropout_rate", type=float, default=0.1)

    # 阈值选择（可选）
    p.add_argument("--select_thr_fpr_max", type=float, default=0.10, help="在 val 上选择满足该 FPR 上限、且 MCC 最大的阈值")
    p.add_argument("--thr_grid_n", type=int, default=101)

    p.add_argument("--out_json", type=str, default=str(default_out))
    p.add_argument("--out_tab_csv", type=str, default=str(default_tab))
    p.add_argument("--out_fig_png", type=str, default=str(default_fig))
    p.add_argument("--out_ckpt", type=str, default=str(default_ckpt))
    p.add_argument("--log_file", type=str, default=None)
    return p.parse_args()


def main():
    _require_torch()
    import torch

    args = parse_args()
    setup_logging(log_file=args.log_file)

    eval_parquet = resolve_path(args.eval_parquet)
    splits_json = resolve_path(args.splits_json)
    cache_dir = resolve_path(args.era5_daily_cache_dir)

    out_json = resolve_path(args.out_json)
    out_tab = resolve_path(args.out_tab_csv)
    out_fig = resolve_path(args.out_fig_png)
    out_ckpt = resolve_path(args.out_ckpt)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_tab.parent.mkdir(parents=True, exist_ok=True)
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)

    # 读取数据（Zone A only）
    need_cols = [
        "event_id",
        "zone_type",
        "flag",
        "window_type",
        "sample_date",
        "grid_i",
        "grid_j",
        *FEATURE_COLUMNS,
    ]
    df = pd.read_parquet(eval_parquet, columns=need_cols)
    df = df[df["zone_type"].astype(int) == 0].copy()
    if df.empty:
        raise ValueError("eval_parquet 中未找到 Zone A（zone_type=0）样本。")

    # ERA5 covariates
    df = attach_era5_covariates(df, cache_dir=cache_dir)

    # Split
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

    # 标准化（仅用 train 拟合）
    x_train_raw = train_df[list(FEATURE_COLUMNS)].astype("float32").to_numpy(copy=True)
    mean, std = _standardize_fit(x_train_raw)

    x_tr, w_tr, y_tr = _build_arrays(train_df, use_weights=bool(args.use_weights), mean=mean, std=std)
    x_va, w_va, y_va = _build_arrays(val_df, use_weights=bool(args.use_weights), mean=mean, std=std)
    x_te, w_te, y_te = _build_arrays(test_df, use_weights=bool(args.use_weights), mean=mean, std=std)

    # Build model（小模型默认）
    from src.config import WEFTTConfig
    from src.models.we_ftt import create_we_ftt_model

    base = dict(WEFTTConfig().BEST_PARAMS)
    base.update(
        {
            "hidden_dim": int(args.hidden_dim),
            "ffn_hidden_dim": int(args.ffn_hidden_dim),
            "n_heads": int(args.n_heads),
            "n_layers": int(args.n_layers),
            "dropout_rate": float(args.dropout_rate),
            "use_cls_token": True,
        }
    )
    if int(args.hidden_dim) % int(args.n_heads) != 0:
        raise ValueError("--hidden_dim 必须能被 --n_heads 整除。")

    model = create_we_ftt_model(
        num_features=len(FEATURE_COLUMNS),
        num_classes=2,
        config=base,
        use_weight_enhancement=bool(args.use_weights),
    )

    device = torch.device("cpu")
    model.to(device)

    # DataLoaders（内存充足时直接 TensorDataset）
    ds_tr = torch.utils.data.TensorDataset(
        torch.from_numpy(x_tr),
        torch.from_numpy(w_tr) if w_tr is not None else torch.empty((len(x_tr), 0), dtype=torch.float32),
        torch.from_numpy(y_tr),
    )
    dl_tr = torch.utils.data.DataLoader(ds_tr, batch_size=int(args.batch_size), shuffle=True, num_workers=0)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )

    best_val_mcc = -1e9
    best_state = None
    patience = max(1, int(args.early_stop_patience))
    patience_cnt = 0
    history: List[Dict[str, float]] = []

    logger.info(
        "训练 WE-FTT（二分类）: n_train=%d n_val=%d epochs=%d batch=%d use_weights=%s",
        len(x_tr),
        len(x_va),
        int(args.epochs),
        int(args.batch_size),
        bool(args.use_weights),
    )

    for epoch in range(int(args.epochs)):
        model.train()
        running = 0.0
        steps = 0
        for xb, wb, yb in dl_tr:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            if bool(args.use_weights):
                out = model(xb, wb.to(device))
            else:
                out = model(xb)
            loss = model.compute_loss(out, yb, epoch=epoch)
            loss.backward()
            optimizer.step()
            running += float(loss.item())
            steps += 1

        tr_loss = running / max(1, steps)

        # val metrics @ thr=0.5
        p_va = _predict_proba_pos(model, x_va, w_va, batch_size=int(args.batch_size))
        yhat_va = (p_va >= 0.5).astype(np.int64)
        m_va = _confusion_metrics(y_va, yhat_va)
        history.append(
            {
                "epoch": float(epoch + 1),
                "train_loss": float(tr_loss),
                "val_mcc": float(m_va.mcc),
                "val_fpr": float(m_va.fpr),
            }
        )
        logger.info(
            "epoch=%d train_loss=%.4f val_mcc=%.4f val_fpr=%.4f (n_val=%d)",
            epoch + 1,
            tr_loss,
            m_va.mcc,
            m_va.fpr,
            m_va.n,
        )

        improved = float(m_va.mcc) > float(best_val_mcc)
        if improved:
            best_val_mcc = float(m_va.mcc)
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                logger.info("early stop at epoch=%d (patience=%d)", epoch + 1, patience)
                break

    if best_state is None:
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    torch.save(
        {
            "version": "ocean_conditioning_weftt_v1",
            "generated_at_utc": datetime.utcnow().isoformat() + "Z",
            "params": {
                "seed": int(args.seed),
                "use_weights": bool(args.use_weights),
                "n_clusters": int(args.n_clusters),
                "epochs": int(args.epochs),
                "batch_size": int(args.batch_size),
                "learning_rate": float(args.learning_rate),
                "weight_decay": float(args.weight_decay),
                "hidden_dim": int(args.hidden_dim),
                "ffn_hidden_dim": int(args.ffn_hidden_dim),
                "n_heads": int(args.n_heads),
                "n_layers": int(args.n_layers),
                "dropout_rate": float(args.dropout_rate),
            },
            "standardize": {"mean": mean.tolist(), "std": std.tolist()},
            "model_state_dict": model.state_dict(),
        },
        out_ckpt,
    )

    # thresholds
    thr_default = 0.5
    thr_grid = np.linspace(0.01, 0.99, int(max(11, args.thr_grid_n))).astype(np.float32)
    p_va = _predict_proba_pos(model, x_va, w_va, batch_size=int(args.batch_size))
    thr_fpr = _pick_threshold_for_fpr(
        y_va,
        p_va,
        fpr_max=float(args.select_thr_fpr_max),
        grid=thr_grid,
    )

    # evaluate on test (before/after mask)
    def _eval_subset(df_sub: pd.DataFrame, *, thr: float) -> Metrics:
        x, w, y = _build_arrays(df_sub, use_weights=bool(args.use_weights), mean=mean, std=std)
        p = _predict_proba_pos(model, x, w, batch_size=int(args.batch_size))
        yhat = (p >= float(thr)).astype(np.int64)
        return _confusion_metrics(y, yhat)

    before_default = _eval_subset(test_df, thr=thr_default)
    masked_df = apply_era5_mask(
        test_df,
        wind_thresh=float(args.wind_thresh),
        precip_thresh=float(args.precip_thresh),
    )
    if bool(args.use_weights) and artifacts:
        masked_df = apply_weighting_artifacts(masked_df, artifacts)
    after_default = _eval_subset(masked_df, thr=thr_default)

    before_thr_fpr = None
    after_thr_fpr = None
    if thr_fpr is not None:
        before_thr_fpr = _eval_subset(test_df, thr=float(thr_fpr))
        after_thr_fpr = _eval_subset(masked_df, thr=float(thr_fpr))

    # write table
    row = {
        "subset": "zone_a_test",
        "wind_thresh_mps": float(args.wind_thresh),
        "precip_thresh_mm_day": float(args.precip_thresh),
        "use_weights": bool(args.use_weights),
        "n_before": int(before_default.n),
        "n_after": int(after_default.n),
        "thr_default": float(thr_default),
        "fpr_before": float(before_default.fpr),
        "fpr_before_ci_low": float(before_default.fpr_ci_low),
        "fpr_before_ci_high": float(before_default.fpr_ci_high),
        "mcc_before": float(before_default.mcc),
        "fpr_after": float(after_default.fpr),
        "fpr_after_ci_low": float(after_default.fpr_ci_low),
        "fpr_after_ci_high": float(after_default.fpr_ci_high),
        "mcc_after": float(after_default.mcc),
    }
    if thr_fpr is not None and before_thr_fpr is not None and after_thr_fpr is not None:
        row.update(
            {
                "thr_val_selected_fpr_max": float(args.select_thr_fpr_max),
                "thr_val_selected": float(thr_fpr),
                "fpr_before_thr_selected": float(before_thr_fpr.fpr),
                "mcc_before_thr_selected": float(before_thr_fpr.mcc),
                "fpr_after_thr_selected": float(after_thr_fpr.fpr),
                "mcc_after_thr_selected": float(after_thr_fpr.mcc),
            }
        )

    pd.DataFrame([row]).to_csv(out_tab, index=False)

    # figure
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(5.5, 3.2))
        xs = ["Before", "After"]
        ys = [before_default.fpr, after_default.fpr]
        yerr = [
            [before_default.fpr - before_default.fpr_ci_low, after_default.fpr - after_default.fpr_ci_low],
            [before_default.fpr_ci_high - before_default.fpr, after_default.fpr_ci_high - after_default.fpr],
        ]
        plt.bar(xs, ys, yerr=yerr, capsize=5)
        plt.ylim(0, 1)
        plt.ylabel("FPR (Zone A)")
        plt.title("ERA5 Hard Mask Conditioning (WE-FTT)")
        plt.tight_layout()
        plt.savefig(out_fig, dpi=200)
        plt.close()
    except Exception as e:
        logger.warning("绘图失败：%s", e)

    payload = {
        "version": "ocean_conditioning_weftt_v1",
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
            "wind_thresh": float(args.wind_thresh),
            "precip_thresh": float(args.precip_thresh),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "hidden_dim": int(args.hidden_dim),
            "ffn_hidden_dim": int(args.ffn_hidden_dim),
            "n_heads": int(args.n_heads),
            "n_layers": int(args.n_layers),
            "dropout_rate": float(args.dropout_rate),
            "thr_default": float(thr_default),
            "thr_val_selected_fpr_max": float(args.select_thr_fpr_max),
            "thr_val_selected": (None if thr_fpr is None else float(thr_fpr)),
        },
        "standardize": {"mean": mean.tolist(), "std": std.tolist()},
        "weighting": artifacts_to_jsonable(artifacts),
        "train_history": history,
        "test": {
            "before_default": before_default.__dict__,
            "after_default": after_default.__dict__,
            "before_thr_selected": (None if before_thr_fpr is None else before_thr_fpr.__dict__),
            "after_thr_selected": (None if after_thr_fpr is None else after_thr_fpr.__dict__),
            "n_masked": int(len(masked_df)),
        },
        "artifacts": {"table_csv": str(out_tab), "figure_png": str(out_fig), "checkpoint": str(out_ckpt)},
    }
    write_json(out_json, payload)
    logger.info("写入: %s", out_json)


if __name__ == "__main__":
    main()

