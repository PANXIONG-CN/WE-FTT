from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans


@dataclass(frozen=True)
class FeatureWeightingArtifact:
    feature: str
    n_clusters: int
    cluster_centers: List[float]
    cluster_weights: List[float]


def _binary_label_from_flag(df: pd.DataFrame, *, flag_col: str = "flag") -> np.ndarray:
    if flag_col not in df.columns:
        raise ValueError(f"输入数据缺少 {flag_col} 列（0/1）。")
    y = df[flag_col].astype("int64").to_numpy()
    if not np.isin(y, [0, 1]).all():
        raise ValueError(f"{flag_col} 列必须为 0/1。")
    return y


def _compute_cluster_weights(labels: np.ndarray, y: np.ndarray, n_clusters: int) -> np.ndarray:
    """
    weight[c] = |P(cluster=c|pos) - P(cluster=c|neg)|, 并做 0~1 归一化。
    """
    pos = y == 1
    neg = y == 0
    n_pos = int(np.sum(pos))
    n_neg = int(np.sum(neg))
    if n_pos == 0 or n_neg == 0:
        return np.ones(int(n_clusters), dtype=np.float32)

    w = np.zeros(int(n_clusters), dtype=np.float32)
    for c in range(int(n_clusters)):
        p_pos = float(np.sum(labels[pos] == c)) / n_pos
        p_neg = float(np.sum(labels[neg] == c)) / n_neg
        w[c] = abs(p_pos - p_neg)

    mx = float(np.max(w))
    return (w / mx) if mx > 0 else np.ones(int(n_clusters), dtype=np.float32)


def add_foldwise_kmeans_weights(
    train_df: pd.DataFrame,
    other_dfs: Sequence[pd.DataFrame],
    *,
    feature_columns: Sequence[str],
    n_clusters: int,
    seed: int,
    batch_size: int = 65536,
    flag_col: str = "flag",
    weight_col_suffix: str = "_cluster_labels_weight",
) -> Tuple[pd.DataFrame, List[pd.DataFrame], List[FeatureWeightingArtifact]]:
    """
    仅使用 train_df 拟合离散化与权重映射，然后应用到 other_dfs（val/test/...）。
    """
    y_train = _binary_label_from_flag(train_df, flag_col=flag_col)
    artifacts: List[FeatureWeightingArtifact] = []

    for feature in feature_columns:
        x_train = train_df[feature].astype("float32").to_numpy().reshape(-1, 1)
        model = MiniBatchKMeans(
            n_clusters=int(n_clusters),
            random_state=int(seed),
            batch_size=int(batch_size),
            n_init="auto",
        )
        model.fit(x_train)

        train_labels = model.predict(x_train)
        cluster_w = _compute_cluster_weights(train_labels, y_train, n_clusters=int(n_clusters))

        def apply(df: pd.DataFrame) -> pd.DataFrame:
            x = df[feature].astype("float32").to_numpy().reshape(-1, 1)
            lbl = model.predict(x)
            df[f"{feature}{weight_col_suffix}"] = cluster_w[lbl].astype(np.float32)
            return df

        train_df = apply(train_df)
        out_others = [apply(d) for d in other_dfs]

        centers = [float(x) for x in model.cluster_centers_.reshape(-1)]
        artifacts.append(
            FeatureWeightingArtifact(
                feature=str(feature),
                n_clusters=int(n_clusters),
                cluster_centers=centers,
                cluster_weights=[float(x) for x in cluster_w.reshape(-1)],
            )
        )

    return train_df, out_others, artifacts


def artifacts_to_jsonable(artifacts: Sequence[FeatureWeightingArtifact]):
    return [asdict(a) for a in artifacts]


def apply_weighting_artifacts(
    df: pd.DataFrame,
    artifacts: Sequence[FeatureWeightingArtifact],
    *,
    weight_col_suffix: str = "_cluster_labels_weight",
) -> pd.DataFrame:
    """
    使用“训练折内”生成的 artifacts 为新数据计算权重列（不再拟合/挖掘）。

    说明：
    - 每个 feature 是 1D，簇分配采用最近中心（等价于 1D kmeans 的最近质心规则）
    - 复杂度约为 O(n * k)；k 通常很小（默认 5）
    """
    for a in artifacts:
        if a.feature not in df.columns:
            raise ValueError(f"输入数据缺少特征列: {a.feature}")
        centers = np.asarray(a.cluster_centers, dtype=np.float32).reshape(1, -1)  # (1,k)
        weights = np.asarray(a.cluster_weights, dtype=np.float32).reshape(-1)  # (k,)
        x = df[a.feature].astype("float32").to_numpy().reshape(-1, 1)  # (n,1)
        # (n,k)
        idx = np.argmin(np.abs(x - centers), axis=1)
        df[f"{a.feature}{weight_col_suffix}"] = weights[idx].astype(np.float32)
    return df
