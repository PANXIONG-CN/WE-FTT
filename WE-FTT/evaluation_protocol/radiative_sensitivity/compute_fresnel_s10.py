#!/usr/bin/env python3
"""
T8 半物理辐射传输灵敏度（Tab S10，最小可用实现）。

模型（KISS）：
- 平滑界面 Fresnel 反射率（H/V）
- 地表发射率 e = 1 - R
- 顶层亮温近似：TB_toa = tau * e * Ts + (1 - tau) * Tatm

灵敏度变量：
- 地表等效介电常数 eps_r（通过有限差分估计 dTB/d eps_r）
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

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


_TAU_BY_FREQ: Dict[str, float] = {
    "06": 0.98,
    "10": 0.97,
    "23": 0.93,
    "36": 0.88,
    "89": 0.75,
}


def _parse_csv_floats(text: str) -> List[float]:
    vals: List[float] = []
    for part in str(text).split(","):
        s = part.strip()
        if not s:
            continue
        vals.append(float(s))
    if not vals:
        raise ValueError("参数列表不能为空。")
    return vals


def _feature_meta(feature: str) -> Dict[str, str]:
    # 形如 BT_06_H
    parts = str(feature).split("_")
    if len(parts) != 3:
        raise ValueError(f"无法解析特征名: {feature}")
    return {"freq_code": parts[1], "pol": parts[2]}


def _fresnel_reflectivity(*, eps_r: float, theta_rad: float, pol: str) -> float:
    if eps_r <= 1.0:
        raise ValueError(f"eps_r 必须 > 1，当前={eps_r}")
    s = math.sin(theta_rad)
    c = math.cos(theta_rad)
    root = math.sqrt(max(eps_r - s * s, 1e-12))
    p = str(pol).upper()
    if p == "H":
        gamma = (c - root) / (c + root)
    elif p == "V":
        gamma = (eps_r * c - root) / (eps_r * c + root)
    else:
        raise ValueError(f"未知极化: {pol}")
    r = float(gamma * gamma)
    return min(max(r, 0.0), 1.0)


def _tb_toa_k(*, eps_r: float, theta_rad: float, pol: str, ts_k: float, tatm_k: float, tau: float) -> float:
    r = _fresnel_reflectivity(eps_r=eps_r, theta_rad=theta_rad, pol=pol)
    emiss = 1.0 - r
    return float(tau * emiss * ts_k + (1.0 - tau) * tatm_k)


def _rows(
    *,
    features: Iterable[str],
    theta_deg: float,
    ts_k: float,
    tatm_k: float,
    eps_refs: List[float],
    delta_fracs: List[float],
) -> List[dict]:
    theta_rad = math.radians(float(theta_deg))
    out: List[dict] = []
    for feature in features:
        meta = _feature_meta(feature)
        freq_code = meta["freq_code"]
        pol = meta["pol"]
        tau = float(_TAU_BY_FREQ.get(freq_code, 0.90))

        for eps_ref in eps_refs:
            if eps_ref <= 1.0:
                continue
            tb_ref = _tb_toa_k(
                eps_r=float(eps_ref),
                theta_rad=theta_rad,
                pol=pol,
                ts_k=float(ts_k),
                tatm_k=float(tatm_k),
                tau=tau,
            )
            r_ref = _fresnel_reflectivity(eps_r=float(eps_ref), theta_rad=theta_rad, pol=pol)
            e_ref = 1.0 - r_ref

            for frac in delta_fracs:
                frac = float(frac)
                if frac <= 0:
                    continue
                eps_plus = float(eps_ref) * (1.0 + frac)
                eps_minus = float(eps_ref) * (1.0 - frac)
                if eps_minus <= 1.0:
                    eps_minus = 1.000001
                tb_plus = _tb_toa_k(
                    eps_r=eps_plus,
                    theta_rad=theta_rad,
                    pol=pol,
                    ts_k=float(ts_k),
                    tatm_k=float(tatm_k),
                    tau=tau,
                )
                tb_minus = _tb_toa_k(
                    eps_r=eps_minus,
                    theta_rad=theta_rad,
                    pol=pol,
                    ts_k=float(ts_k),
                    tatm_k=float(tatm_k),
                    tau=tau,
                )
                d_tb_plus = tb_plus - tb_ref
                d_tb_minus = tb_minus - tb_ref
                d_tb_d_eps = (tb_plus - tb_minus) / (eps_plus - eps_minus)
                d_tb_d_ln_eps = float(eps_ref) * d_tb_d_eps
                out.append(
                    {
                        "feature": str(feature),
                        "freq_code": str(freq_code),
                        "pol": str(pol),
                        "variable": "surface_dielectric_eps_r",
                        "theta_deg": float(theta_deg),
                        "surface_temp_k": float(ts_k),
                        "atm_temp_k": float(tatm_k),
                        "tau_band": float(tau),
                        "eps_ref": float(eps_ref),
                        "delta_eps_frac": float(frac),
                        "reflectivity_ref": float(r_ref),
                        "emissivity_ref": float(e_ref),
                        "tb_ref_k": float(tb_ref),
                        "tb_plus_k": float(tb_plus),
                        "tb_minus_k": float(tb_minus),
                        "delta_tb_plus_k": float(d_tb_plus),
                        "delta_tb_minus_k": float(d_tb_minus),
                        "dTB_dEps_k_per_unit": float(d_tb_d_eps),
                        "dTB_dLnEps_k": float(d_tb_d_ln_eps),
                    }
                )
    return out


def parse_args():
    repo = get_repo_paths()
    default_csv = repo.eval_root / "radiative_sensitivity" / "tables" / "tab_s10_fresnel_sensitivity_v1.csv"
    default_meta = repo.eval_root / "radiative_sensitivity" / "results" / "fresnel_sensitivity_v1.meta.json"

    p = argparse.ArgumentParser(description="Compute Fresnel sensitivity table for Tab S10 (minimal)")
    p.add_argument("--theta_deg", type=float, default=55.0)
    p.add_argument("--surface_temp_k", type=float, default=300.0)
    p.add_argument("--atm_temp_k", type=float, default=260.0)
    p.add_argument("--eps_refs", type=str, default="4,10,25", help="逗号分隔，例如 4,10,25")
    p.add_argument("--delta_eps_fracs", type=str, default="0.01,0.05", help="逗号分隔，例如 0.01,0.05")
    p.add_argument("--out_csv", type=str, default=str(default_csv))
    p.add_argument("--out_meta_json", type=str, default=str(default_meta))
    p.add_argument("--log_file", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging(log_file=args.log_file)

    out_csv = resolve_path(args.out_csv)
    out_meta = resolve_path(args.out_meta_json)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_meta.parent.mkdir(parents=True, exist_ok=True)

    eps_refs = _parse_csv_floats(args.eps_refs)
    delta_fracs = _parse_csv_floats(args.delta_eps_fracs)

    rows = _rows(
        features=FEATURE_COLUMNS,
        theta_deg=float(args.theta_deg),
        ts_k=float(args.surface_temp_k),
        tatm_k=float(args.atm_temp_k),
        eps_refs=eps_refs,
        delta_fracs=delta_fracs,
    )
    if not rows:
        raise ValueError("未生成任何行，请检查 eps_refs/delta_eps_fracs 参数。")

    df = pd.DataFrame(rows).sort_values(
        by=["feature", "eps_ref", "delta_eps_frac"],
        kind="mergesort",
    )
    df.to_csv(out_csv, index=False)

    write_json(
        out_meta,
        {
            "version": "fresnel_sensitivity_v1",
            "generated_at_utc": datetime.utcnow().isoformat() + "Z",
            "out_csv": str(out_csv),
            "params": {
                "theta_deg": float(args.theta_deg),
                "surface_temp_k": float(args.surface_temp_k),
                "atm_temp_k": float(args.atm_temp_k),
                "eps_refs": [float(x) for x in eps_refs],
                "delta_eps_fracs": [float(x) for x in delta_fracs],
                "tau_by_freq": _TAU_BY_FREQ,
            },
            "rows": int(len(df)),
            "notes": {
                "model": "TB_toa = tau*(1-R)*Ts + (1-tau)*Tatm, R from Fresnel",
                "sensitivity": "finite difference around eps_ref with +/- delta_eps_frac",
            },
        },
    )
    logger.info("写入: %s", out_csv)


if __name__ == "__main__":
    main()

