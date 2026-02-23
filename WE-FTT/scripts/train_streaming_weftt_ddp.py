#!/usr/bin/env python3
"""
DDP streaming trainer for WE-FTT/FT-Transformer on multi-GPU (NVLink-friendly).

Features
- Keeps current model structure from src.models.we_ftt (WE-FTT / FT-Transformer)
- Streaming Parquet reading via IterableDataset (no full-data pandas copies)
- Two-pass global mean/std (rank0 only) + broadcast
- Deterministic hash-based split (train/val/test) + rank shard (no overlap)
- AMP (optional), gradient accumulation, optional SyncBatchNorm

Usage (two GPUs, mp.spawn)
  python scripts/train_streaming_weftt_ddp.py \
    --model_name we_ftt \
    --data_path /path/to/training_dataset.parquet \
    --epochs 50 --batch_size 2048 --learning_rate 1e-3 \
    --compute_norm --sync_bn --amp

You may also set NVLink-friendly NCCL env vars before launch, e.g.:
  export CUDA_VISIBLE_DEVICES=0,1
  export NCCL_P2P_LEVEL=NVL
  export NCCL_IB_DISABLE=1
"""

import os
import sys
import math
import json
import argparse
import logging
import multiprocessing as mp
from collections import deque
from pathlib import Path
from typing import Dict, Any, Optional, Iterator, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp_spawn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import IterableDataset, DataLoader

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.config import WEFTTConfig
from src.models.we_ftt import create_we_ftt_model, create_ft_transformer_model
from src.utils import setup_logging, set_random_seeds


logger = logging.getLogger(__name__)


def stable_hash_u01(idx: np.ndarray, seed: int) -> np.ndarray:
    """Deterministic 32-bit LCG hash → [0,1) uniform float."""
    h = (idx.astype(np.uint64) * np.uint64(1103515245) + np.uint64(seed)) & np.uint64(0xFFFFFFFF)
    return (h.astype(np.float64) / float(0x100000000)).astype(np.float32)


def compute_feature_stats_parquet(path: str, feature_cols: list, batch_rows: int = 1_000_000) -> Tuple[np.ndarray, np.ndarray, int]:
    """Two-pass stats (mean/std) using Welford accumulation over Parquet by batches."""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    n_feat = len(feature_cols)
    count = 0
    mean = np.zeros(n_feat, dtype=np.float64)
    M2 = np.zeros(n_feat, dtype=np.float64)

    for batch in pf.iter_batches(columns=feature_cols, batch_size=batch_rows):
        cols = [batch.column(i).to_numpy(zero_copy_only=False).astype(np.float64, copy=False) for i in range(n_feat)]
        X = np.column_stack(cols)
        count_b = X.shape[0]
        if count_b == 0:
            continue
        new_count = count + count_b
        delta = X.mean(axis=0) - mean
        mean += delta * (count_b / new_count)
        diffs = X - mean
        M2 += (diffs ** 2).sum(axis=0)
        count = new_count

    var = M2 / max(1, count - 1)
    std = np.sqrt(np.maximum(var, 1e-12))
    return mean.astype(np.float32), std.astype(np.float32), count


def compute_class_counts_parquet(path: str, num_classes: int, batch_rows: int = 1_000_000) -> np.ndarray:
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    counts = np.zeros(num_classes, dtype=np.int64)
    for batch in pf.iter_batches(columns=['label'], batch_size=batch_rows):
        arr = batch.column(0).to_numpy(zero_copy_only=False)
        if arr.dtype.kind == 'f':
            arr = arr.astype(np.int64, copy=False)
        elif arr.dtype.kind in ('U', 'S', 'O'):
            arr = arr.astype(np.int64, copy=False)
        else:
            arr = arr.astype(np.int64, copy=False)
        binc = np.bincount(arr, minlength=num_classes)
        counts[:len(binc)] += binc[:num_classes]
    return counts


class ParquetSplitStreamingDataset(IterableDataset):
    def __init__(
        self,
        path: str,
        feature_cols: list,
        weight_cols: Optional[list],
        label_col: str,
        split: str,  # 'train'|'val'|'test'
        test_size: float = 0.2,
        val_size: float = 0.2,
        seed: int = 42,
        batch_rows: int = 1_000_000,
        shuffle_buffer: int = 100_000,
        norm_mean: Optional[np.ndarray] = None,
        norm_std: Optional[np.ndarray] = None,
        world_size: int = 1,
        rank: int = 0,
        max_rows: Optional[int] = None,
        target_probs: Optional[np.ndarray] = None,
        num_classes: Optional[int] = None,
    ):
        super().__init__()
        assert split in {'train', 'val', 'test'}
        self.path = path
        self.feature_cols = feature_cols
        self.weight_cols = weight_cols or []
        self.label_col = label_col
        self.split = split
        self.test_size = float(test_size)
        self.val_size = float(val_size)
        self.seed = int(seed)
        self.batch_rows = int(batch_rows)
        self.shuffle_buffer = int(shuffle_buffer) if split == 'train' else 0
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.world_size = int(world_size)
        self.rank = int(rank)
        self.max_rows = max_rows
        self.target_probs = target_probs
        self.num_classes = num_classes
        self.do_stratified = target_probs is not None and num_classes is not None and split == 'train'
        self._epoch = 0
        self._epoch_shared = mp.Value('i', 0)

    def set_epoch(self, epoch: int):
        self._epoch = int(epoch)
        self._epoch_shared.value = int(epoch)

    def _split_mask(self, start_idx: int, n: int) -> np.ndarray:
        idx = np.arange(start_idx, start_idx + n, dtype=np.int64)
        u = stable_hash_u01(idx, seed=self.seed)  # split independent of epoch
        th_test = self.test_size
        th_val = self.test_size + self.val_size
        if self.split == 'test':
            return u < th_test
        elif self.split == 'val':
            return (u >= th_test) & (u < th_val)
        else:
            return u >= th_val

    def _rank_mask(self, start_idx: int, n: int) -> np.ndarray:
        if self.world_size <= 1:
            return np.ones(n, dtype=bool)
        idx = np.arange(start_idx, start_idx + n, dtype=np.int64)
        return (idx % self.world_size) == self.rank

    def __iter__(self) -> Iterator[Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]]:
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(self.path)
        n_feat = len(self.feature_cols)
        n_w = len(self.weight_cols)

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        else:
            worker_id = 0
            num_workers = 1

        epoch = self._epoch_shared.value
        rng = np.random.RandomState(self.seed + epoch * 131 + self.rank * 17 + worker_id * 97)

        if self.do_stratified:
            assert self.num_classes is not None
            per_class_feat = [deque() for _ in range(self.num_classes)]
            per_class_weight = [deque() for _ in range(self.num_classes)] if n_w > 0 else None
            total_buffered = 0
            threshold = max(65536, self.shuffle_buffer if self.shuffle_buffer > 0 else 0)
        else:
            buf_feat = []
            buf_weight = [] if n_w > 0 else None
            buf_label = []
            threshold = max(65536, self.shuffle_buffer if self.shuffle_buffer > 0 else 0)

        emitted = 0
        row_base = 0

        for batch in pf.iter_batches(columns=self.feature_cols + self.weight_cols + [self.label_col], batch_size=self.batch_rows):
            batch_len = batch.num_rows
            if batch_len == 0:
                row_base += batch_len
                continue

            m_split = self._split_mask(row_base, batch_len)
            m_rank = self._rank_mask(row_base, batch_len)
            if num_workers > 1:
                idx = np.arange(row_base, row_base + batch_len, dtype=np.int64)
                m_worker = (idx % num_workers) == worker_id
                mask = m_split & m_rank & m_worker
            else:
                mask = m_split & m_rank
            row_base += batch_len
            if not mask.any():
                continue

            cols = [batch.column(i).to_numpy(zero_copy_only=False).astype(np.float32, copy=False) for i in range(n_feat)]
            X = np.column_stack(cols)[mask]
            if self.norm_mean is not None and self.norm_std is not None:
                X = (X - self.norm_mean) / self.norm_std

            W = None
            if n_w > 0:
                wcols = [batch.column(n_feat + i).to_numpy(zero_copy_only=False).astype(np.float32, copy=False) for i in range(n_w)]
                W = np.column_stack(wcols)[mask]

            y_arr = batch.column(n_feat + n_w).to_numpy(zero_copy_only=False)
            if y_arr.dtype.kind == 'f':
                y = y_arr.astype(np.int64, copy=False)
            elif y_arr.dtype.kind in ('U', 'S', 'O'):
                y = y_arr.astype(np.int64, copy=False)
            else:
                y = y_arr.astype(np.int64, copy=False)
            y = y[mask]

            if self.do_stratified:
                for i in range(X.shape[0]):
                    cls = int(y[i])
                    if 0 <= cls < self.num_classes:
                        per_class_feat[cls].append(X[i].copy())
                        if n_w > 0:
                            per_class_weight[cls].append(W[i].copy())
                        total_buffered += 1
                while total_buffered >= threshold and self.target_probs is not None:
                    cls = int(rng.choice(self.num_classes, p=self.target_probs))
                    if not per_class_feat[cls]:
                        found = False
                        for alt in rng.permutation(self.num_classes):
                            if per_class_feat[alt]:
                                cls = int(alt)
                                found = True
                                break
                        if not found:
                            break
                    xf = per_class_feat[cls].popleft()
                    wf = per_class_weight[cls].popleft() if n_w > 0 else None
                    total_buffered -= 1
                    yield xf, wf, np.int64(cls)
                    emitted += 1
                    if self.max_rows and emitted >= self.max_rows:
                        return
            else:
                for i in range(X.shape[0]):
                    Wi = (W[i] if W is not None else None)
                    if self.shuffle_buffer > 0:
                        buf_feat.append(X[i:i+1].copy())
                        buf_label.append(y[i:i+1].copy())
                        if n_w > 0:
                            buf_weight.append(W[i:i+1].copy())
                        if sum(arr.shape[0] for arr in buf_feat) >= threshold:
                            Xb = np.concatenate(buf_feat, axis=0)
                            yb = np.concatenate(buf_label, axis=0)
                            Wb = np.concatenate(buf_weight, axis=0) if n_w > 0 else None
                            idx = rng.permutation(Xb.shape[0])
                            Xb = Xb[idx]; yb = yb[idx]
                            if Wb is not None:
                                Wb = Wb[idx]
                            for j in range(Xb.shape[0]):
                                yield Xb[j], (Wb[j] if Wb is not None else None), yb[j]
                                emitted += 1
                                if self.max_rows and emitted >= self.max_rows:
                                    return
                            buf_feat, buf_label = [], []
                            if n_w > 0:
                                buf_weight = []
                    else:
                        yield X[i], Wi, y[i]
                        emitted += 1
                        if self.max_rows and emitted >= self.max_rows:
                            return

        # flush remaining
        if self.do_stratified:
            if n_w > 0:
                for cls in range(self.num_classes):
                    while per_class_feat[cls]:
                        xf = per_class_feat[cls].popleft()
                        wf = per_class_weight[cls].popleft()
                        yield xf, wf, np.int64(cls)
                        emitted += 1
                        if self.max_rows and emitted >= self.max_rows:
                            return
            else:
                for cls in range(self.num_classes):
                    while per_class_feat[cls]:
                        xf = per_class_feat[cls].popleft()
                        yield xf, None, np.int64(cls)
                        emitted += 1
                        if self.max_rows and emitted >= self.max_rows:
                            return
        else:
            if self.split == 'train' and buf_feat:
                Xb = np.concatenate(buf_feat, axis=0)
                yb = np.concatenate(buf_label, axis=0)
                Wb = np.concatenate(buf_weight, axis=0) if (buf_weight and len(buf_weight) > 0) else None
                idx = rng.permutation(Xb.shape[0])
                Xb = Xb[idx]; yb = yb[idx]
                if Wb is not None:
                    Wb = Wb[idx]
                for i in range(Xb.shape[0]):
                    yield Xb[i], (Wb[i] if Wb is not None else None), yb[i]
                    emitted += 1
                    if self.max_rows and emitted >= self.max_rows:
                        return


def collate_stream(batch):
    xs = [torch.from_numpy(b[0]) for b in batch]
    ws = [torch.from_numpy(b[1]) for b in batch if b[1] is not None]
    ys = [torch.tensor(b[2], dtype=torch.long) for b in batch]
    X = torch.stack(xs, dim=0)
    W = torch.stack(ws, dim=0) if len(ws) == len(batch) else None
    y = torch.stack(ys, dim=0)
    return X, W, y


def parse_args():
    p = argparse.ArgumentParser(description='DDP streaming training for WE-FTT/FT-Transformer')
    p.add_argument('--model_name', type=str, required=True, choices=['we_ftt', 'ft_transformer'])
    p.add_argument('--data_path', type=str, required=True)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch_size', type=int, default=2048, help='per-GPU batch size')
    p.add_argument('--learning_rate', type=float, default=1e-3)
    p.add_argument('--random_seed', type=int, default=42)
    p.add_argument('--output_dir', type=str, default='results/streaming_ddp')
    p.add_argument('--test_size', type=float, default=0.2)
    p.add_argument('--val_size', type=float, default=0.2)
    p.add_argument('--buffer', type=int, default=200_000, help='train shuffle buffer per rank')
    p.add_argument('--rowgroup_rows', type=int, default=1_000_000)
    p.add_argument('--max_val_rows', type=int, default=5_000_000)
    p.add_argument('--max_test_rows', type=int, default=5_000_000)
    p.add_argument('--compute_norm', action='store_true')
    p.add_argument('--world_size', type=int, default=None, help='defaults to torch.cuda.device_count()')
    p.add_argument('--sync_bn', action='store_true', help='convert model to SyncBatchNorm')
    p.add_argument('--amp', action='store_true', help='enable mixed precision (AMP)')
    p.add_argument('--grad_accum', type=int, default=1, help='gradient accumulation steps')
    p.add_argument('--num_workers', type=int, default=0, help='number of workers for training DataLoader')
    p.add_argument('--eval_workers', type=int, default=0, help='number of workers for val/test DataLoader')
    p.add_argument('--stratified', action='store_true', help='enable stratified streaming for training data')
    p.add_argument('--stratified_mode', type=str, choices=['data', 'uniform'], default='data', help='target distribution for stratified streaming')
    return p.parse_args()


def setup_dist(rank: int, world_size: int):
    os.environ.setdefault('MASTER_ADDR', 'localhost')
    os.environ.setdefault('MASTER_PORT', '23456')
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_dist():
    if dist.is_initialized():
        dist.destroy_process_group()


def main_worker(rank: int, world_size: int, args):
    is_main = (rank == 0)
    if is_main:
        setup_logging()
    setup_dist(rank, world_size)

    # Device & seed
    device = torch.device(f'cuda:{rank}')
    set_random_seeds(args.random_seed + rank)

    # Config columns
    base_cfg = WEFTTConfig()
    feature_cols = base_cfg.COLUMNS_FEATURES
    weight_cols = base_cfg.COLUMNS_WEIGHTS
    label_col = base_cfg.LABEL_COLUMN
    n_feat = len(feature_cols)

    # Global norm (rank0 compute, broadcast)
    norm_mean = None
    norm_std = None
    if args.compute_norm and args.data_path.endswith('.parquet'):
        if is_main:
            logger.info('Computing global feature mean/std (two-pass, rank0)...')
            m, s, n = compute_feature_stats_parquet(args.data_path, feature_cols, batch_rows=args.rowgroup_rows)
            logger.info(f'Global stats computed on {n} rows')
            t_m = torch.from_numpy(m).to(device)
            t_s = torch.from_numpy(s).to(device)
        else:
            t_m = torch.empty(n_feat, dtype=torch.float32, device=device)
            t_s = torch.empty(n_feat, dtype=torch.float32, device=device)
        dist.broadcast(t_m, src=0)
        dist.broadcast(t_s, src=0)
        norm_mean = t_m.cpu().numpy()
        norm_std = t_s.cpu().numpy()
    dist.barrier()

    # Stratified target probabilities (optional)
    target_probs = None
    num_classes = 10  # dataset-specific
    if args.stratified:
        if is_main:
            logger.info('Computing global class counts (rank0)...')
            counts = compute_class_counts_parquet(args.data_path, num_classes=num_classes, batch_rows=args.rowgroup_rows)
            if args.stratified_mode == 'uniform':
                probs = np.ones(num_classes, dtype=np.float32) / num_classes
            else:
                total = counts.sum()
                probs = counts.astype(np.float32) / float(total if total > 0 else 1)
            t_probs = torch.from_numpy(probs).to(device)
        else:
            t_probs = torch.empty(num_classes, dtype=torch.float32, device=device)
        dist.broadcast(t_probs, src=0)
        target_probs = t_probs.cpu().numpy()
    dist.barrier()

    # Build model
    num_classes = 10  # align with your dataset
    if args.model_name == 'we_ftt':
        model = create_we_ftt_model(num_features=len(feature_cols), num_classes=num_classes, config=base_cfg.BEST_PARAMS, use_weight_enhancement=True)
    else:
        model = create_ft_transformer_model(num_features=len(feature_cols), num_classes=num_classes, config=base_cfg.BEST_PARAMS)
    model.to(device)
    if args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    ddp_model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)

    # Optimizer & AMP
    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    grad_accum = max(1, int(args.grad_accum))

    # Datasets per-rank
    train_ds = ParquetSplitStreamingDataset(
        path=args.data_path,
        feature_cols=feature_cols,
        weight_cols=weight_cols,
        label_col=label_col,
        split='train',
        test_size=args.test_size,
        val_size=args.val_size,
        seed=args.random_seed,
        batch_rows=args.rowgroup_rows,
        shuffle_buffer=args.buffer,
        norm_mean=norm_mean,
        norm_std=norm_std,
        world_size=world_size,
        rank=rank,
        max_rows=None,
        target_probs=target_probs,
        num_classes=num_classes,
    )
    val_ds = ParquetSplitStreamingDataset(
        path=args.data_path,
        feature_cols=feature_cols,
        weight_cols=weight_cols,
        label_col=label_col,
        split='val',
        test_size=args.test_size,
        val_size=args.val_size,
        seed=args.random_seed,
        batch_rows=args.rowgroup_rows,
        shuffle_buffer=0,
        norm_mean=norm_mean,
        norm_std=norm_std,
        world_size=world_size,
        rank=rank,
        max_rows=int(math.ceil(args.max_val_rows / world_size)) if args.max_val_rows else None,
        target_probs=None,
        num_classes=num_classes,
    )
    test_ds = ParquetSplitStreamingDataset(
        path=args.data_path,
        feature_cols=feature_cols,
        weight_cols=weight_cols,
        label_col=label_col,
        split='test',
        test_size=args.test_size,
        val_size=args.val_size,
        seed=args.random_seed,
        batch_rows=args.rowgroup_rows,
        shuffle_buffer=0,
        norm_mean=norm_mean,
        norm_std=norm_std,
        world_size=world_size,
        rank=rank,
        max_rows=int(math.ceil(args.max_test_rows / world_size)) if args.max_test_rows else None,
        target_probs=None,
        num_classes=num_classes,
    )

    # DataLoaders (worker>0 would need per-worker partitioning; keep 0 for safety)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        collate_fn=collate_stream,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        collate_fn=collate_stream,
        num_workers=args.eval_workers,
        pin_memory=True,
        persistent_workers=args.eval_workers > 0,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        collate_fn=collate_stream,
        num_workers=args.eval_workers,
        pin_memory=True,
        persistent_workers=args.eval_workers > 0,
    )

    best_val = float('inf')
    patience = 5
    patience_cnt = 0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(args.epochs):
        # epoch seed for train shuffling (per rank)
        train_ds.set_epoch(epoch)

        ddp_model.train()
        running = 0.0
        steps = 0
        optimizer.zero_grad(set_to_none=True)
        for step, (xb, wb, yb) in enumerate(train_loader, start=1):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=args.amp):
                if wb is not None:
                    wb = wb.to(device, non_blocking=True)
                    out = ddp_model(xb, wb)
                else:
                    out = ddp_model(xb)
                loss = ddp_model.module.compute_loss(out, yb, epoch)
                if grad_accum > 1:
                    loss = loss / grad_accum
            scaler.scale(loss).backward()

            if step % grad_accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            running += float(loss.item())
            steps += 1
        # all-reduce train loss average for logging
        t_loss = torch.tensor([running / max(1, steps)], dtype=torch.float32, device=device)
        dist.all_reduce(t_loss, op=dist.ReduceOp.SUM)
        t_loss = (t_loss / world_size).item()

        # Validation (sum-reduce losses and counts)
        ddp_model.eval()
        loss_sum = torch.tensor([0.0], dtype=torch.float32, device=device)
        correct = torch.tensor([0.0], dtype=torch.float32, device=device)
        total = torch.tensor([0.0], dtype=torch.float32, device=device)
        with torch.no_grad():
            for xb, wb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=args.amp):
                    if wb is not None:
                        wb = wb.to(device, non_blocking=True)
                        out = ddp_model(xb, wb)
                    else:
                        out = ddp_model(xb)
                    loss = ddp_model.module.compute_loss(out, yb)
                # convert mean loss to sum by multiplying batch size
                bs = torch.tensor([yb.size(0)], dtype=torch.float32, device=device)
                loss_sum += loss * bs
                pred = out.argmax(dim=1)
                correct += (pred == yb).sum().float()
                total += bs
        # reduce across ranks
        for t in (loss_sum, correct, total):
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
        val_loss = (loss_sum / torch.clamp_min(total, 1.0)).item()
        val_acc = (correct / torch.clamp_min(total, 1.0)).item()

        if is_main:
            history['train_loss'].append(t_loss)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            logger.info(f"Epoch {epoch+1}/{args.epochs} - train_loss {t_loss:.4f} - val_loss {val_loss:.4f} - val_acc {val_acc:.4f}")

        improved = (val_loss < best_val)
        if improved:
            best_val = val_loss
            patience_cnt = 0
            if is_main:
                out_dir = Path(args.output_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                torch.save({'model_state_dict': ddp_model.module.state_dict(), 'epoch': epoch+1}, out_dir / 'best_model.pth')
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                if is_main:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                break

    # Test (sum-reduce)
    ddp_model.eval()
    correct = torch.tensor([0.0], dtype=torch.float32, device=device)
    total = torch.tensor([0.0], dtype=torch.float32, device=device)
    with torch.no_grad():
        for xb, wb, yb in test_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=args.amp):
                if wb is not None:
                    wb = wb.to(device, non_blocking=True)
                    out = ddp_model(xb, wb)
                else:
                    out = ddp_model(xb)
            pred = out.argmax(dim=1)
            correct += (pred == yb).sum().float()
            total += torch.tensor([yb.size(0)], dtype=torch.float32, device=device)
    dist.all_reduce(correct, op=dist.ReduceOp.SUM)
    dist.all_reduce(total, op=dist.ReduceOp.SUM)
    test_acc = (correct / torch.clamp_min(total, 1.0)).item()
    if is_main:
        logger.info(f"Test accuracy: {test_acc:.4f} on ~{int(total.item())} examples")
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / 'results.json', 'w') as f:
            json.dump({'history': history, 'test_acc': test_acc}, f, indent=2)

    cleanup_dist()


def main():
    args = parse_args()
    world_size = args.world_size or torch.cuda.device_count()
    assert world_size >= 1, 'No CUDA devices found.'
    mp_spawn.spawn(main_worker, args=(world_size, args), nprocs=world_size, join=True)


if __name__ == '__main__':
    main()
