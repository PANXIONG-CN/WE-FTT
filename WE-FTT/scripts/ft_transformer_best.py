import os
import math
import json
import logging
import copy
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    cohen_kappa_score,
    matthews_corrcoef,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
)
from sklearn.preprocessing import LabelEncoder
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from typing import Optional, List, Dict, Any



class Config:
    BATCH_SIZE = 100000
    NUM_WORKERS = 1
    MEMORY_LIMIT_GB = 100
    NUM_EPOCHS = 20
    PATIENCE = 5
    NUM_TRIALS = 10
    FILE_PATH = "/mnt/hdd_4tb_data/ArchivedWorks/MBT/FTT/updated_code/training_dataset.parquet"
    COLUMNS_FEATURES = [
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
    COLUMNS_WEIGHTS = [
        "BT_06_H_cluster_labels_weight",
        "BT_06_V_cluster_labels_weight",
        "BT_10_H_cluster_labels_weight",
        "BT_10_V_cluster_labels_weight",
        "BT_23_H_cluster_labels_weight",
        "BT_23_V_cluster_labels_weight",
        "BT_36_H_cluster_labels_weight",
        "BT_36_V_cluster_labels_weight",
        "BT_89_H_cluster_labels_weight",
        "BT_89_V_cluster_labels_weight",
    ]
    LABEL_COLUMN = "label"


BEST_PARAMS: Dict[str, Any] = {
    "num_heads": 8,
    "input_embed_dim": 128,
    "num_attn_blocks": 7,
    "attn_dropout": 0.4576631512298116,
    "head_dim": 64,
    "fusion_hidden_dim": 1024,
    "fusion_activation": "silu",
    "learning_rate": 0.0024495117577774,
    "weight_decay": 2.935610188880024e-06,
    "beta1": 0.89615445716568,
    "beta2": 0.9506890081734285,
    "warmup_steps": 287,
    "lr_decay_factor": 0.1937428197419404,
    "layer_norm_eps": 1.051619437458352e-06,
    "gradient_clip_val": 2.0546703848672093,
    "residual_dropout": 0.2899580866406258,
    "residual_scale": 0.7507200933159254,
    "focal_gamma": 0.7438012181829819,
    "focal_momentum": 0.5740236627753215,
}


class ParquetDataset(Dataset):
    def __init__(self, parquet_path: str, feature_cols: list, weight_cols: list, label_col: str, device: str = "cpu"):

        df = pd.read_parquet(parquet_path)

        features_data_list = []
        for col in feature_cols:
            features_data_list.append(df[col].astype("float16").to_numpy())
        self.features = torch.FloatTensor(np.column_stack(features_data_list))

        weights_data_list = []
        for col in weight_cols:
            weights_data_list.append(df[col].astype("float16").to_numpy())
        self.weights = torch.FloatTensor(np.column_stack(weights_data_list))

        self.labels = torch.LongTensor(df[label_col].astype("int64").to_numpy())

        self.to(device)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.weights[idx], self.labels[idx]

    def to(self, device: str):
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.features = self.features.to(device)
        self.weights = self.weights.to(device)
        self.labels = self.labels.to(device)
        self.device = device
        return self


def load_data(parquet_path, config, test_size=0.2, random_state=42):
    dataset = ParquetDataset(
        parquet_path=parquet_path,
        feature_cols=config.COLUMNS_FEATURES,
        weight_cols=config.COLUMNS_WEIGHTS,
        label_col=config.LABEL_COLUMN,
        device="cpu",
    )

    total_size = len(dataset)
    train_size = int((1 - test_size) * total_size)
    test_size = total_size - train_size

    indices = torch.randperm(total_size)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    return train_dataset, test_dataset


def get_sinusoidal_encoding_table(seq_len, embed_dim):
    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / embed_dim)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(embed_dim)]

    sinusoidal_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(seq_len)])
    sinusoidal_table[:, 0::2] = np.sin(sinusoidal_table[:, 0::2])
    sinusoidal_table[:, 1::2] = np.cos(sinusoidal_table[:, 1::2])
    return torch.FloatTensor(sinusoidal_table).unsqueeze(0)


class FTTransformerConfig:
    def __init__(self, task, num_attn_blocks, attn_dropout, num_heads, input_embed_dim, num_classes):
        self.task = task
        self.num_attn_blocks = num_attn_blocks
        self.attn_dropout = attn_dropout
        self.num_heads = num_heads
        self.input_embed_dim = input_embed_dim
        self.num_classes = num_classes


class CustomFTTransformerModel(nn.Module):
    def __init__(
        self,
        config,
        columns_features,
        columns_weights,
        head_dim,
        fusion_hidden_dim,
        fusion_activation,
        layer_norm_eps,
        residual_dropout,
        residual_scale,
    ):
        super().__init__()
        self.feature_proj = nn.Linear(len(columns_features), config.input_embed_dim)
        self.weight_proj = nn.Linear(len(columns_weights), config.input_embed_dim)
        self.layer_norm = nn.LayerNorm(config.input_embed_dim, eps=layer_norm_eps)
        self.attn_blocks = nn.ModuleList(
            [MultiHeadedAttention(config.num_heads, config.input_embed_dim, head_dim, config.attn_dropout) for _ in range(config.num_attn_blocks)]
        )
        self.residual_layers = nn.ModuleList([nn.Linear(config.input_embed_dim, config.input_embed_dim) for _ in range(config.num_attn_blocks)])
        self.feature_fusion = nn.Sequential(
            nn.Linear(config.input_embed_dim, fusion_hidden_dim),
            self._get_activation(fusion_activation),
            nn.Dropout(residual_dropout),
            nn.Linear(fusion_hidden_dim, config.input_embed_dim),
        )
        self.residual_scale = residual_scale
        self._head = nn.Linear(config.input_embed_dim, config.num_classes)

        self.pos_enc = nn.Parameter(get_sinusoidal_encoding_table(seq_len=1, embed_dim=config.input_embed_dim), requires_grad=False)

    def _get_activation(self, activation_name):
        if activation_name == "relu":
            return nn.ReLU()
        elif activation_name == "gelu":
            return nn.GELU()
        elif activation_name == "silu":
            return nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation: {activation_name}")

    def forward(self, features, weights):
        feature_embed = self.feature_proj(features)
        weight_embed = self.weight_proj(weights)
        x = feature_embed * weight_embed
        x = x.unsqueeze(1)
        x = x + self.pos_enc
        x = self.layer_norm(x)

        for attn_block, residual_layer in zip(self.attn_blocks, self.residual_layers):
            residual = x
            x = attn_block(x)
            x = residual_layer(x)
            x = residual * self.residual_scale + x
            x = self.layer_norm(x)
            x = self.feature_fusion(x)

        x = x.squeeze(1)
        logits = self._head(x)
        probs = F.softmax(logits, dim=1)
        return {"logits": logits, "probs": probs}


class MultiHeadedAttention(nn.Module):
    def __init__(self, num_heads, embed_dim, head_dim, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = head_dim
        self.qkv_proj = nn.Linear(embed_dim, 3 * num_heads * head_dim)
        self.o_proj = nn.Linear(num_heads * head_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, seq_length, embed_dim = x.size()
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_length, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, v)
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(batch_size, seq_length, self.num_heads * self.head_dim)
        context = self.o_proj(context)
        return context


class DynamicFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction="mean", momentum=0.9):
        super().__init__()
        self.base_alpha = alpha
        self.current_alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.momentum = momentum
        self.epoch_stats = []

    def update_alpha(self, epoch_stats):
        if self.base_alpha is None:
            return

        self.epoch_stats.append(epoch_stats)
        if len(self.epoch_stats) > 5:
            recent_stats = np.array(self.epoch_stats[-5:])
            class_accuracies = recent_stats.mean(axis=0)

            new_alpha = 1 - class_accuracies
            new_alpha = new_alpha / new_alpha.sum()

            if self.current_alpha is None:
                self.current_alpha = new_alpha
            else:
                self.current_alpha = self.momentum * self.current_alpha + (1 - self.momentum) * new_alpha

    def forward(self, inputs, targets):
        alpha = torch.FloatTensor(self.current_alpha).to(inputs.device) if self.current_alpha is not None else None
        return self._focal_loss(inputs, targets, alpha)

    def _focal_loss(self, inputs, targets, alpha):
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_weight = (1 - pt) ** self.gamma

        if alpha is not None:
            alpha_t = alpha[targets]
            focal_weight = alpha_t * focal_weight

        loss = focal_weight * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class WarmupCosineSchedule:
    def __init__(self, optimizer, warmup_steps, total_steps, lr_decay_factor):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.lr_decay_factor = lr_decay_factor
        self.current_step = 0

        for param_group in self.optimizer.param_groups:
            param_group["initial_lr"] = param_group["lr"]

    def step(self):
        self.current_step += 1
        if self.current_step < self.warmup_steps:
            lr_scale = float(self.current_step) / float(max(1, self.warmup_steps))
        else:
            progress = float(self.current_step - self.warmup_steps) / float(max(1, self.total_steps - self.warmup_steps))
            lr_scale = max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress))) * self.lr_decay_factor

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = param_group["initial_lr"] * lr_scale


class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self):
        train_dataset, test_dataset = load_data(self.config.FILE_PATH, self.config)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=True,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=True,
        )

        results = self.train_with_params(BEST_PARAMS, train_loader, test_loader)
        self.save_results(results)

    def _build_model(self, params: Dict[str, Any]) -> nn.Module:
        model_config = FTTransformerConfig(
            task="multiclass",
            num_attn_blocks=params["num_attn_blocks"],
            attn_dropout=params["attn_dropout"],
            num_heads=params["num_heads"],
            input_embed_dim=params["input_embed_dim"],
            num_classes=10,
        )

        model = CustomFTTransformerModel(
            config=model_config,
            columns_features=Config.COLUMNS_FEATURES,
            columns_weights=Config.COLUMNS_WEIGHTS,
            head_dim=params["head_dim"],
            fusion_hidden_dim=params["fusion_hidden_dim"],
            fusion_activation=params["fusion_activation"],
            layer_norm_eps=params["layer_norm_eps"],
            residual_dropout=params["residual_dropout"],
            residual_scale=params["residual_scale"],
        )
        return model.to(self.device)

    def _compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_probs: np.ndarray) -> Dict[str, Any]:
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=1)
        accuracy = accuracy_score(y_true, y_pred)
        cohen_kappa = cohen_kappa_score(y_true, y_pred)
        mcc = matthews_corrcoef(y_true, y_pred)
        conf_matrix = confusion_matrix(y_true, y_pred)

        metrics: Dict[str, Any] = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "cohen_kappa": float(cohen_kappa),
            "mcc": float(mcc),
            "confusion_matrix": conf_matrix.tolist(),
        }

        class_mapping = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 0, 6: 1, 7: 2, 8: 3, 9: 4}
        y_true_mapped = np.array([class_mapping[label] for label in y_true])
        y_pred_mapped = np.array([class_mapping[label] for label in y_pred])

        for cls in range(5):
            cls_mask = y_true_mapped == cls
            y_true_cls = y_true[cls_mask]
            y_pred_cls = y_pred[cls_mask]
            y_true_cls_binary = np.where(y_true_cls == cls * 2, 0, 1)
            y_pred_cls_binary = np.where(y_pred_cls == cls * 2, 0, 1)

            precision_cls, recall_cls, f1_cls, _ = precision_recall_fscore_support(
                y_true_cls_binary, y_pred_cls_binary, average="binary", zero_division=1
            )
            conf_matrix_cls = confusion_matrix(y_true_cls_binary, y_pred_cls_binary).tolist()

            if len(np.unique(y_true_cls_binary)) >= 2:
                auc_cls = float(roc_auc_score(y_true_cls_binary, y_probs[cls_mask, cls * 2 + 1]))
                ap_cls = float(average_precision_score(y_true_cls_binary, y_probs[cls_mask, cls * 2 + 1]))
            else:
                auc_cls = float("nan")
                ap_cls = float("nan")

            metrics[f"class_{cls}"] = {
                "precision": float(precision_cls),
                "recall": float(recall_cls),
                "f1": float(f1_cls),
                "auc": auc_cls,
                "ap": ap_cls,
                "confusion_matrix": conf_matrix_cls,
            }

        return metrics

    def _evaluate(self, model: nn.Module, data_loader: DataLoader) -> Dict[str, Any]:
        model.eval()
        y_pred, y_true, y_probs = [], [], []
        with torch.no_grad():
            for features_batch, weights_batch, y_batch in data_loader:
                features_batch = features_batch.to(self.device)
                weights_batch = weights_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                outputs = model(features_batch, weights_batch)
                _, predicted = torch.max(outputs["logits"].data, 1)
                y_pred.extend(predicted.cpu().numpy())
                y_true.extend(y_batch.cpu().numpy())
                y_probs.extend(outputs["probs"].cpu().numpy())
        return self._compute_metrics(np.array(y_true), np.array(y_pred), np.array(y_probs))

    def train_with_params(self, params: Dict[str, Any], train_loader: DataLoader, test_loader: DataLoader) -> Dict[str, Any]:
        model = self._build_model(params)
        criterion = DynamicFocalLoss(alpha=None, gamma=params["focal_gamma"], reduction="mean", momentum=params["focal_momentum"])
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=params["learning_rate"],
            weight_decay=params["weight_decay"],
            betas=(params["beta1"], params["beta2"]),
        )
        total_steps = self.config.NUM_EPOCHS * len(train_loader)
        scheduler = WarmupCosineSchedule(
            optimizer,
            warmup_steps=params["warmup_steps"],
            total_steps=total_steps,
            lr_decay_factor=params["lr_decay_factor"],
        )
        scaler = GradScaler()

        history = []
        best_mcc = -float("inf")
        best_state = None
        best_epoch_metrics = None
        patience_counter = 0

        for epoch in range(self.config.NUM_EPOCHS):
            model.train()
            total_loss = 0.0
            for features_batch, weights_batch, y_batch in train_loader:
                features_batch = features_batch.to(self.device)
                weights_batch = weights_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                with autocast(enabled=torch.cuda.is_available()):
                    logits = model(features_batch, weights_batch)["logits"]
                    loss = criterion(logits, y_batch)

                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), params["gradient_clip_val"])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                total_loss += float(loss.item())

            avg_loss = total_loss / max(1, len(train_loader))
            metrics = self._evaluate(model, test_loader)
            metrics["epoch"] = epoch + 1
            metrics["train_loss"] = avg_loss
            history.append(metrics)

            current_mcc = metrics["mcc"]
            if current_mcc > best_mcc or best_state is None:
                best_mcc = current_mcc
                patience_counter = 0
                best_state = copy.deepcopy(model.state_dict())
                best_epoch_metrics = metrics
            else:
                patience_counter += 1
                if patience_counter >= self.config.PATIENCE:
                    logging.getLogger(__name__).info(f"Early stopping at epoch {epoch + 1}")
                    break

        if best_state is None:
            best_state = copy.deepcopy(model.state_dict())
            best_epoch_metrics = history[-1]

        out_dir = Path("results/ft_transformer_best")
        out_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = out_dir / "model_best.pth"
        torch.save({
            "state_dict": best_state,
            "params": params,
            "best_metrics": best_epoch_metrics,
        }, checkpoint_path)

        model.load_state_dict(best_state)
        test_metrics = self._evaluate(model, test_loader)

        return {
            "history": history,
            "best_metrics": best_epoch_metrics,
            "test_metrics": test_metrics,
            "checkpoint": str(checkpoint_path),
            "params": params,
        }

    def save_results(self, results: Dict[str, Any]):
        out_dir = Path("results/ft_transformer_best")
        out_dir.mkdir(parents=True, exist_ok=True)
        results_path = out_dir / "metrics.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)



def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    config = Config()
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
