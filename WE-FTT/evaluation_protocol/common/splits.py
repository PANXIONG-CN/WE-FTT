from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Set, Tuple

import pandas as pd


@dataclass(frozen=True)
class EventSplits:
    train_event_ids: Set[str]
    val_event_ids: Set[str]
    test_event_ids: Set[str]


def load_event_splits(path: str | Path) -> EventSplits:
    p = Path(path)
    payload = json.loads(p.read_text(encoding="utf-8"))
    train = set(payload.get("train_event_ids", []))
    val = set(payload.get("val_event_ids", []))
    test = set(payload.get("test_event_ids", []))
    if not train or not val or not test:
        raise ValueError("event splits 文件缺少 train/val/test event_ids。")
    return EventSplits(train_event_ids=train, val_event_ids=val, test_event_ids=test)


def split_df_by_event_id(
    df: pd.DataFrame,
    splits: EventSplits,
    *,
    event_id_col: str = "event_id",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if event_id_col not in df.columns:
        raise ValueError(f"输入数据缺少 {event_id_col} 列，无法事件级切分。")
    ev = df[event_id_col].astype(str)
    m_train = ev.isin(splits.train_event_ids)
    m_val = ev.isin(splits.val_event_ids)
    m_test = ev.isin(splits.test_event_ids)
    if (m_train & m_val).any() or (m_train & m_test).any() or (m_val & m_test).any():
        raise ValueError("同一 event_id 同时落入多个 split（splits 不一致）。")
    return df[m_train].copy(), df[m_val].copy(), df[m_test].copy()

