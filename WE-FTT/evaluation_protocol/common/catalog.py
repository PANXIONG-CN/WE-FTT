from __future__ import annotations

import csv
import hashlib
import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EarthquakeEvent:
    event_id: str
    event_date: date
    latitude: float
    longitude: float
    depth_km: float
    magnitude: float
    zone_type: Optional[int]
    place: str
    time_str: str


def _stable_event_id(row: dict) -> str:
    """
    当目录缺少 `id` 时，生成稳定ID（不依赖行号）。
    """
    key = "|".join(
        str(row.get(k, "")) for k in ("date", "time", "latitude", "longitude", "depth", "mag", "place")
    )
    h = hashlib.sha1(key.encode("utf-8", errors="replace")).hexdigest()
    return f"EV_{h[:16]}"


def _parse_date(s: str) -> Optional[date]:
    s = (s or "").strip()
    try:
        # 期望 YYYY-MM-DD
        y, m, d = s.split("-")
        return date(int(y), int(m), int(d))
    except Exception:
        return None


def iter_events(
    catalog_csv_path: str | Path,
    *,
    min_mag: float = 7.0,
    zones: Optional[Sequence[int]] = None,
    require_zone: bool = True,
    marine_zone_type: int = 0,
    marine_depth_max_km: float = 70.0,
) -> Iterator[EarthquakeEvent]:
    """
    从 `earthquake_catalog.csv` 读取并筛选事件。

    说明：
    - 本仓库目录的 `time` 字段存在两种格式：`HH:MM:SS` 与 `MM:SS.s`（后者缺少小时）。
      evaluation_protocol 的核心逻辑以“日粒度”工作，不依赖精确时刻，因此仅保留原始 time_str。
    """
    p = Path(catalog_csv_path)
    if not p.exists():
        raise FileNotFoundError(f"地震目录不存在: {p}")

    zones_set = set(zones) if zones else None

    with p.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                mag = float(row.get("mag", "nan"))
            except Exception:
                continue
            if not (mag >= float(min_mag)):
                continue

            d = _parse_date(row.get("date", ""))
            if d is None:
                continue

            try:
                lat = float(row.get("latitude", "nan"))
                lon = float(row.get("longitude", "nan"))
                dep = float(row.get("depth", "nan"))
            except Exception:
                continue

            zone_raw = (row.get("Type_", "") or "").strip()
            zone_type: Optional[int]
            if zone_raw == "":
                zone_type = None
            else:
                try:
                    zone_type = int(zone_raw)
                except Exception:
                    zone_type = None

            if require_zone and zone_type is None:
                continue
            if zones_set is not None and zone_type is not None and zone_type not in zones_set:
                continue

            # Zone A（海洋）仅取浅源（与论文设定一致）
            if zone_type == marine_zone_type and dep >= float(marine_depth_max_km):
                continue

            eid = (row.get("id", "") or "").strip()
            if not eid:
                eid = _stable_event_id(row)

            yield EarthquakeEvent(
                event_id=eid,
                event_date=d,
                latitude=lat,
                longitude=lon,
                depth_km=dep,
                magnitude=mag,
                zone_type=zone_type,
                place=(row.get("place", "") or "").strip(),
                time_str=(row.get("time", "") or "").strip(),
            )


def load_events(
    catalog_csv_path: str | Path,
    *,
    min_mag: float = 7.0,
    zones: Optional[Sequence[int]] = None,
    require_zone: bool = True,
    marine_zone_type: int = 0,
    marine_depth_max_km: float = 70.0,
) -> list[EarthquakeEvent]:
    events = list(
        iter_events(
            catalog_csv_path,
            min_mag=min_mag,
            zones=zones,
            require_zone=require_zone,
            marine_zone_type=marine_zone_type,
            marine_depth_max_km=marine_depth_max_km,
        )
    )
    # 去重：以 event_id 为主
    seen = set()
    uniq: list[EarthquakeEvent] = []
    for e in events:
        if e.event_id in seen:
            continue
        seen.add(e.event_id)
        uniq.append(e)
    if len(uniq) != len(events):
        logger.info("事件去重: %d -> %d", len(events), len(uniq))
    return uniq

