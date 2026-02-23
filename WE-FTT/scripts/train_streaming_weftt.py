#!/usr/bin/env python3
"""
Memory-efficient streaming trainer for WE-FTT/FT-Transformer.

Key ideas:
- Keep current model structure (WE-FTT / FT-Transformer from src.models).
- Avoid full-data pandas preprocessing (drop_duplicates/IQR/StandardScaler copies).
- Do two-pass Parquet statistics (optional) + streaming batches via IterableDataset.
- Split train/val/test by deterministic hash on global row index (approx stratified for large N).
- Optional shuffle buffer for train to approximate global shuffle.

Requirements: pyarrow
"""

import os
import sys
import math
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Iterator, Tuple

import numpy as np
import torch
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
    # h = (a * idx + seed) % 2**32
    h = (idx.astype(np.uint64) * np.uint64(1103515245) + np.uint64(seed)) & np.uint64(0xFFFFFFFF)
    return (h.astype(np.float64) / float(0x100000000)).astype(np.float32)


def compute_feature_stats_parquet(path: str, feature_cols: list, batch_rows: int = 1_000_000) -> Tuple[np.ndarray, np.ndarray, int]:
    """Two-pass stats (mean/std) using Welford-like accumulation over Parquet by batches.
    Returns: (mean[features], std[features], total_rows)
    """
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    n_feat = len(feature_cols)
    count = 0
    mean = np.zeros(n_feat, dtype=np.float64)
    M2 = np.zeros(n_feat, dtype=np.float64)

    for batch in pf.iter_batches(columns=feature_cols, batch_size=batch_rows):
        # stack features (batch_len, n_feat) as float64 for stable stats
        cols = [batch.column(i).to_numpy(zero_copy_only=False).astype(np.float64, copy=False) for i in range(n_feat)]
        X = np.column_stack(cols)
        # Welford
        count_b = X.shape[0]
        if count_b == 0:
            continue
        new_count = count + count_b
        delta = X.mean(axis=0) - mean
        mean += delta * (count_b / new_count)
        # sum of squares of differences
        diffs = X - mean
        M2 += (diffs ** 2).sum(axis=0)
        count = new_count

    # Avoid zero std
    var = M2 / max(1, count - 1)
    std = np.sqrt(np.maximum(var, 1e-12))
    return mean.astype(np.float32), std.astype(np.float32), count


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
        max_rows: Optional[int] = None,  # cap rows for this split (e.g., validation)
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
        self.max_rows = max_rows

        self._epoch = 0

    def set_epoch(self, epoch: int):
        self._epoch = int(epoch)

    def _belongs_mask(self, start_idx: int, n: int) -> np.ndarray:
        idx = np.arange(start_idx, start_idx + n, dtype=np.int64)
        u = stable_hash_u01(idx, seed=self.seed + self._epoch)  # epoch-dependent shuffling
        th_test = self.test_size
        th_val = self.test_size + self.val_size
        if self.split == 'test':
            return u < th_test
        elif self.split == 'val':
            return (u >= th_test) & (u < th_val)
        else:
            return u >= th_val

    def __iter__(self) -> Iterator[Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]]:
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(self.path)
        n_feat = len(self.feature_cols)
        n_w = len(self.weight_cols)

        # shuffle buffer (train only)
        buf_feat = []
        buf_weight = [] if n_w > 0 else None
        buf_label = []

        emitted = 0
        row_base = 0

        for batch in pf.iter_batches(columns=self.feature_cols + self.weight_cols + [self.label_col], batch_size=self.batch_rows):
            batch_len = batch.num_rows
            mask = self._belongs_mask(row_base, batch_len)
            row_base += batch_len
            if not mask.any():
                continue

            # Slice columns
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
                # encode strings to ints via numpy unique on the fly (costly). Assume numeric here.
                y = y_arr.astype(np.int64, copy=False)
            else:
                y = y_arr.astype(np.int64, copy=False)
            y = y[mask]

            if self.split == 'train' and self.shuffle_buffer > 0:
                buf_feat.append(X)
                buf_label.append(y)
                if n_w > 0:
                    buf_weight.append(W)
                # Flatten buffer when exceeding capacity
                if sum(arr.shape[0] for arr in buf_feat) >= self.shuffle_buffer:
                    Xb = np.concatenate(buf_feat, axis=0)
                    yb = np.concatenate(buf_label, axis=0)
                    Wb = np.concatenate(buf_weight, axis=0) if n_w > 0 else None
                    # Random shuffle
                    idx = np.random.RandomState(self.seed + self._epoch).permutation(Xb.shape[0])
                    Xb = Xb[idx]
                    yb = yb[idx]
                    if Wb is not None:
                        Wb = Wb[idx]
                    # Emit in chunks ~64k for throughput
                    step = 65536
                    for s in range(0, Xb.shape[0], step):
                        Xe = Xb[s:s+step]
                        ye = yb[s:s+step]
                        We = Wb[s:s+step] if Wb is not None else None
                        for i in range(Xe.shape[0]):
                            yield Xe[i], (We[i] if We is not None else None), ye[i]
                            emitted += 1
                            if self.max_rows and emitted >= self.max_rows:
                                return
                    buf_feat, buf_label = [], []
                    if n_w > 0:
                        buf_weight = []
            else:
                # Direct emit without big buffer (val/test)
                for i in range(X.shape[0]):
                    Wi = (W[i] if W is not None else None)
                    yield X[i], Wi, y[i]
                    emitted += 1
                    if self.max_rows and emitted >= self.max_rows:
                        return

        # Flush remaining buffer
        if self.split == 'train' and buf_feat:
            Xb = np.concatenate(buf_feat, axis=0)
            yb = np.concatenate(buf_label, axis=0)
            Wb = np.concatenate(buf_weight, axis=0) if (buf_weight and len(buf_weight) > 0) else None
            idx = np.random.RandomState(self.seed + self._epoch).permutation(Xb.shape[0])
            Xb = Xb[idx]; yb = yb[idx]
            if Wb is not None:
                Wb = Wb[idx]
            for i in range(Xb.shape[0]):
                yield Xb[i], (Wb[i] if Wb is not None else None), yb[i]
                emitted += 1
                if self.max_rows and emitted >= self.max_rows:
                    return


def collate_stream(batch):
    # batch is list of tuples (x, w or None, y)
    xs = [torch.from_numpy(b[0]) for b in batch]
    ws = [torch.from_numpy(b[1]) for b in batch if b[1] is not None]
    ys = [torch.tensor(b[2], dtype=torch.long) for b in batch]
    X = torch.stack(xs, dim=0)
    W = torch.stack(ws, dim=0) if len(ws) == len(batch) else None
    y = torch.stack(ys, dim=0)
    return X, W, y


def parse_args():
    p = argparse.ArgumentParser(description='Streaming training for WE-FTT/FT-Transformer')
    p.add_argument('--model_name', type=str, required=True, choices=['we_ftt', 'ft_transformer'])
    p.add_argument('--data_path', type=str, required=True, help='Parquet/CSV file (Parquet recommended)')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch_size', type=int, default=32768)
    p.add_argument('--learning_rate', type=float, default=1e-3)
    p.add_argument('--random_seed', type=int, default=42)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--output_dir', type=str, default='results/streaming')
    p.add_argument('--test_size', type=float, default=0.2)
    p.add_argument('--val_size', type=float, default=0.2)
    p.add_argument('--buffer', type=int, default=100_000, help='Train shuffle buffer')
    p.add_argument('--rowgroup_rows', type=int, default=1_000_000)
    p.add_argument('--max_val_rows', type=int, default=5_000_000)
    p.add_argument('--max_test_rows', type=int, default=5_000_000)
    p.add_argument('--compute_norm', action='store_true', help='Compute global mean/std and z-normalize features')
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging()

    # Device
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Config columns
    base_cfg = WEFTTConfig()
    feature_cols = base_cfg.COLUMNS_FEATURES
    weight_cols = base_cfg.COLUMNS_WEIGHTS
    label_col = base_cfg.LABEL_COLUMN

    # Random seed
    set_random_seeds(args.random_seed)

    # Optional normalization (two-pass over Parquet)
    norm_mean = None
    norm_std = None
    if args.compute_norm and args.data_path.endswith('.parquet'):
        logger.info('Computing global feature mean/std (two-pass)...')
        norm_mean, norm_std, nrows = compute_feature_stats_parquet(args.data_path, feature_cols, batch_rows=args.rowgroup_rows)
        logger.info(f'Global stats computed on {nrows} rows')

    # Build model
    num_classes = 10  # align with your 10-class dataset
    if args.model_name == 'we_ftt':
        model = create_we_ftt_model(num_features=len(feature_cols), num_classes=num_classes, config=base_cfg.BEST_PARAMS, use_weight_enhancement=True)
    else:
        model = create_ft_transformer_model(num_features=len(feature_cols), num_classes=num_classes, config=base_cfg.BEST_PARAMS)
    model.to(device)

    # Optimizer (AdamW)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)

    # Datasets (streaming)
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
        max_rows=None,
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
        max_rows=args.max_val_rows,
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
        max_rows=args.max_test_rows,
    )

    # Loaders
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, collate_fn=collate_stream, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=collate_stream, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, collate_fn=collate_stream, num_workers=0)

    # Train loops
    best_val = float('inf')
    patience = 5
    patience_cnt = 0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(args.epochs):
        # Update epoch for dataset hashing/shuffle
        train_ds.set_epoch(epoch)

        model.train()
        running = 0.0
        steps = 0
        for xb, wb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            if wb is not None:
                wb = wb.to(device)
                out = model(xb, wb)
            else:
                out = model(xb)
            loss = model.compute_loss(out, yb, epoch)
            loss.backward()
            optimizer.step()
            running += float(loss.item())
            steps += 1
        tr_loss = running / max(1, steps)

        # Validation (streamed, capped by max_val_rows)
        model.eval()
        v_loss = 0.0
        v_corr = 0
        v_total = 0
        with torch.no_grad():
            for xb, wb, yb in val_loader:
                xb = xb.to(device); yb = yb.to(device)
                if wb is not None:
                    wb = wb.to(device)
                    out = model(xb, wb)
                else:
                    out = model(xb)
                loss = model.compute_loss(out, yb)
                v_loss += float(loss.item())
                pred = out.argmax(dim=1)
                v_corr += int((pred == yb).sum().item())
                v_total += int(yb.size(0))
        val_loss = v_loss / max(1, math.ceil(v_total / max(1, args.batch_size)))
        val_acc = (v_corr / v_total) if v_total > 0 else 0.0

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        logger.info(f"Epoch {epoch+1}/{args.epochs} - train_loss {tr_loss:.4f} - val_loss {val_loss:.4f} - val_acc {val_acc:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            patience_cnt = 0
            # Save best checkpoint (state_dict only for brevity)
            out_dir = Path(args.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch+1}, out_dir / 'best_model.pth')
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    # Test pass (streamed, capped)
    model.eval()
    t_corr = 0; t_total = 0
    with torch.no_grad():
        for xb, wb, yb in test_loader:
            xb = xb.to(device); yb = yb.to(device)
            if wb is not None:
                wb = wb.to(device)
                out = model(xb, wb)
            else:
                out = model(xb)
            pred = out.argmax(dim=1)
            t_corr += int((pred == yb).sum().item())
            t_total += int(yb.size(0))
    test_acc = (t_corr / t_total) if t_total > 0 else 0.0
    logger.info(f"Test accuracy: {test_acc:.4f} on {t_total} examples")

    # Save history/results
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'results.json', 'w') as f:
        json.dump({'history': history, 'test_acc': test_acc}, f, indent=2)


if __name__ == '__main__':
    main()

