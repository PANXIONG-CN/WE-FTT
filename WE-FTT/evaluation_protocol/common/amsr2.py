from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

from .geo import GRID_N_LAT, GRID_N_LON


logger = logging.getLogger(__name__)


FEATURE_COLUMNS = [
    "BT_06_H",
    "BT_06_V",
    "BT_10_H",
    "BT_10_V",
    "BT_23_H",
    "BT_23_V",
    "BT_36_H",
    "BT_36_V",
    "BT_89_H",
    "BT_89_V",
]


@dataclass(frozen=True)
class AMSR2Spec:
    """
    AMSR2-L3 读取约定（与原始目录结构对齐）。
    """
    orbit_tag: str = "EQMD"  # Descending（夜间）
    mean_tag: str = "01D"    # 日均
    product_suffix: str = "LA2220220"


def _freq_dir(freq_code: str) -> str:
    return f"{freq_code}-25"


def amsr2_daily_file(amsr2_root: str | Path, freq_code: str, day: date, spec: AMSR2Spec) -> Path:
    root = Path(amsr2_root)
    y = f"{day.year:04d}"
    m = f"{day.month:02d}"
    fname = f"GW1AM2_{day.strftime('%Y%m%d')}_{spec.mean_tag}_{spec.orbit_tag}_L3SGT{freq_code}{spec.product_suffix}.h5"
    return root / _freq_dir(freq_code) / y / m / fname


def _require_h5py():
    try:
        import h5py  # noqa: F401
    except Exception as e:
        raise ImportError(
            "缺少依赖 h5py。请在你的运行环境中安装后重试（建议创建独立venv）。"
        ) from e


def _read_scale_factor(ds) -> float:
    try:
        # HDF5 属性名包含空格：SCALE FACTOR
        v = ds.attrs.get("SCALE FACTOR", None)
        if v is None:
            return 0.01
        # 可能是 numpy 数组
        if hasattr(v, "__len__"):
            return float(v[0])
        return float(v)
    except Exception:
        return 0.01


def _decode_tb(raw: np.ndarray, scale: float) -> np.ndarray:
    """
    原始数据为 uint16。观测无效值在高纬/缺测处常见为 65534/65535。
    """
    raw_u = raw.astype(np.uint16, copy=False)
    invalid = raw_u >= np.uint16(65533)
    out = raw_u.astype(np.float32) * float(scale)
    out[invalid] = np.nan
    return out


def _read_points_from_dataset(ds, i: np.ndarray, j: np.ndarray) -> np.ndarray:
    """
    从 2D 网格读取散点值（优先尝试点选择；失败则回退到局部切片）。
    """
    # 先尝试向量化点读取（h5py 支持时最省IO）
    try:
        vals = ds[i, j]
        return np.asarray(vals)
    except Exception:
        pass

    # 回退：读取覆盖这些点的最小切片，再索引
    i_min, i_max = int(i.min()), int(i.max())
    j_min, j_max = int(j.min()), int(j.max())
    block_area = (i_max - i_min + 1) * (j_max - j_min + 1)
    if block_area <= 512 * 512:
        block = ds[i_min : i_max + 1, j_min : j_max + 1]
        block = np.asarray(block)
        return block[i - i_min, j - j_min]

    # 大范围散点：改用“瓦片读取”避免逐点 ds[i,j] 极慢（尤其在跨日界线/跨半球时）。
    # 经验上 256×256 足够小，可控内存，同时能显著减少 HDF5 随机读开销。
    tile_h = 256
    tile_w = 256
    i64 = i.astype(np.int64, copy=False)
    j64 = j.astype(np.int64, copy=False)
    ti = (i64 // tile_h).astype(np.int64, copy=False)
    tj = (j64 // tile_w).astype(np.int64, copy=False)

    # 组合成一维 key 以便排序分组（ti<=2, tj<=5）
    key = ti * 100 + tj
    order = np.argsort(key, kind="mergesort")
    key_s = key[order]

    out = np.empty(i64.shape[0], dtype=ds.dtype)
    n = int(i64.shape[0])
    start = 0
    n_lat = int(ds.shape[0])
    n_lon = int(ds.shape[1])

    while start < n:
        k = int(key_s[start])
        end = start + 1
        while end < n and int(key_s[end]) == k:
            end += 1

        idxs = order[start:end]
        # 同组点共享一个 tile
        ti0 = int(ti[idxs[0]])
        tj0 = int(tj[idxs[0]])
        i0 = ti0 * tile_h
        j0 = tj0 * tile_w
        i1 = min(i0 + tile_h, n_lat)
        j1 = min(j0 + tile_w, n_lon)

        block = np.asarray(ds[i0:i1, j0:j1])
        out[idxs] = block[i64[idxs] - i0, j64[idxs] - j0]

        start = end

    return out


class AMSR2DailyGrid:
    """
    读取某一天的 5 个频段（H/V）TB 网格，并支持按像元索引批量取值。
    """

    def __init__(self, amsr2_root: str | Path, day: date, *, spec: Optional[AMSR2Spec] = None):
        _require_h5py()
        import h5py  # type: ignore

        self.amsr2_root = Path(amsr2_root)
        self.day = day
        self.spec = spec or AMSR2Spec()
        self._h5: Dict[str, "h5py.File"] = {}
        self._ds: Dict[str, Tuple[object, object]] = {}
        self._scale: Dict[str, Tuple[float, float]] = {}

    def open(self) -> "AMSR2DailyGrid":
        import h5py  # type: ignore

        for freq in ("06", "10", "23", "36", "89"):
            fp = amsr2_daily_file(self.amsr2_root, freq, self.day, self.spec)
            if not fp.exists():
                raise FileNotFoundError(f"AMSR2 日文件不存在: {fp}")
            try:
                f = h5py.File(fp, "r")
            except Exception as e:
                raise OSError(f"无法打开 AMSR2 HDF5: {fp}") from e
            self._h5[freq] = f
            try:
                ds_h = f["/Brightness Temperature (H)"]
                ds_v = f["/Brightness Temperature (V)"]
            except Exception as e:
                try:
                    f.close()
                except Exception:
                    pass
                raise KeyError(f"AMSR2 HDF5 缺少 TB 数据集: {fp}") from e
            self._ds[freq] = (ds_h, ds_v)
            self._scale[freq] = (_read_scale_factor(ds_h), _read_scale_factor(ds_v))
        return self

    def close(self) -> None:
        for f in self._h5.values():
            try:
                f.close()
            except Exception:
                pass
        self._h5.clear()
        self._ds.clear()
        self._scale.clear()

    def __enter__(self) -> "AMSR2DailyGrid":
        return self.open()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def read_features_at(self, i: np.ndarray, j: np.ndarray) -> Dict[str, np.ndarray]:
        """
        输入 i/j 为同长度的一维数组，返回 10 个特征列（float32，K），无效为 NaN。
        """
        if i.ndim != 1 or j.ndim != 1 or i.shape[0] != j.shape[0]:
            raise ValueError("i/j 必须为等长一维数组")
        if i.size == 0:
            return {k: np.array([], dtype=np.float32) for k in FEATURE_COLUMNS}
        if int(i.min()) < 0 or int(i.max()) >= GRID_N_LAT or int(j.min()) < 0 or int(j.max()) >= GRID_N_LON:
            raise ValueError("i/j 超出网格范围")

        out: Dict[str, np.ndarray] = {}
        for freq in ("06", "10", "23", "36", "89"):
            ds_h, ds_v = self._ds[freq]
            s_h, s_v = self._scale[freq]
            raw_h = _read_points_from_dataset(ds_h, i, j)
            raw_v = _read_points_from_dataset(ds_v, i, j)
            tb_h = _decode_tb(raw_h, s_h)
            tb_v = _decode_tb(raw_v, s_v)
            out[f"BT_{freq}_H"] = tb_h.astype(np.float32, copy=False)
            out[f"BT_{freq}_V"] = tb_v.astype(np.float32, copy=False)
        return out
