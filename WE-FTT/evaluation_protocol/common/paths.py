from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class RepoPaths:
    repo_root: Path
    weftt_root: Path
    eval_root: Path


def get_repo_paths() -> RepoPaths:
    """
    返回本仓库路径约定：
    - repo_root: /.../MBT/Final
    - weftt_root: repo_root/WE-FTT
    - eval_root: weftt_root/evaluation_protocol
    """
    here = Path(__file__).resolve()
    eval_root = here.parents[1]  # evaluation_protocol/common -> evaluation_protocol
    weftt_root = here.parents[2]  # .../WE-FTT
    repo_root = here.parents[3]  # .../Final
    return RepoPaths(repo_root=repo_root, weftt_root=weftt_root, eval_root=eval_root)


def resolve_path(path: str | Path, base: Optional[Path] = None) -> Path:
    """
    将字符串路径解析为 Path：
    - 绝对路径：原样返回
    - 相对路径：以 base（缺省为 weftt_root）为基准

    兼容约定：
    - 允许传入以 "WE-FTT/" 开头的仓库相对路径（此时以 repo_root 为基准）
      例如：WE-FTT/data/raw/earthquake_catalog.csv
    """
    p = Path(path)
    if p.is_absolute():
        return p

    if base is None:
        # 允许 "./WE-FTT/..." 与 "WE-FTT/..." 两种写法
        parts = list(p.parts)
        while parts and parts[0] == ".":
            parts = parts[1:]
        if parts and parts[0] == "WE-FTT":
            repo_root = get_repo_paths().repo_root
            return (repo_root / Path(*parts)).resolve()

    base_dir = base or get_repo_paths().weftt_root
    return (base_dir / p).resolve()
