import os
import torch
import numpy as np
import pandas as pd
import optuna
import math
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler, Subset
from torch.nn.parallel import DistributedDataParallel as DDP
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
from torch.optim import AdamW
from torch.cuda.amp import GradScaler
import matplotlib.pyplot as plt
import h5py
import pyarrow.parquet as pq
from typing import Optional, List


class Config:
    BATCH_SIZE = 100000
    NUM_WORKERS = 1
    MEMORY_LIMIT_GB = 100
    NUM_EPOCHS = 20
    PATIENCE = 5
    NUM_TRIALS = 10
    FILE_PATH = "/home/panxiong/MBT/FTT/Mindformers/training_dataset_demo1.parquet"
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
        per_gpu_batch_size = self.config.BATCH_SIZE

        train_dataset, test_dataset = load_data(self.config.FILE_PATH, self.config)

        train_labels = [label for _, _, label in train_dataset]
        class_weights = torch.tensor(np.bincount(train_labels), dtype=torch.float32)
        class_weights = class_weights.max() / class_weights

        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: self.objective(trial, train_dataset, test_dataset, class_weights), n_trials=self.config.NUM_TRIALS)
        self.save_results(study)

    def objective(self, trial, train_dataset, test_dataset, class_weights):
        per_gpu_batch_size = self.config.BATCH_SIZE

        train_loader = DataLoader(train_dataset, batch_size=per_gpu_batch_size, shuffle=True, num_workers=self.config.NUM_WORKERS, pin_memory=False)
        test_loader = DataLoader(test_dataset, batch_size=per_gpu_batch_size, shuffle=False, num_workers=self.config.NUM_WORKERS, pin_memory=False)

        num_heads = trial.suggest_categorical("num_heads", [4, 8, 16, 32])
        input_embed_dim = trial.suggest_categorical("input_embed_dim", [64, 128, 256, 512])
        num_attn_blocks = trial.suggest_int("num_attn_blocks", 2, 8)
        attn_dropout = trial.suggest_float("attn_dropout", 0.1, 0.5)
        head_dim = trial.suggest_categorical("head_dim", [32, 64, 128])
        fusion_hidden_dim = trial.suggest_categorical("fusion_hidden_dim", [128, 256, 512, 1024])
        fusion_activation = trial.suggest_categorical("fusion_activation", ["relu", "gelu", "silu"])
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        beta1 = trial.suggest_float("beta1", 0.8, 0.99)
        beta2 = trial.suggest_float("beta2", 0.9, 0.999)
        warmup_steps = trial.suggest_int("warmup_steps", 100, 1000)
        lr_decay_factor = trial.suggest_float("lr_decay_factor", 0.1, 0.5)
        layer_norm_eps = trial.suggest_float("layer_norm_eps", 1e-7, 1e-5, log=True)
        gradient_clip_val = trial.suggest_float("gradient_clip_val", 0.5, 5.0)
        residual_dropout = trial.suggest_float("residual_dropout", 0.1, 0.5)
        residual_scale = trial.suggest_float("residual_scale", 0.1, 1.0)
        focal_gamma = trial.suggest_float("focal_gamma", 0.5, 5.0)
        focal_momentum = trial.suggest_float("focal_momentum", 0.5, 0.99)

        model_config = FTTransformerConfig(
            task="multiclass",
            num_attn_blocks=num_attn_blocks,
            attn_dropout=attn_dropout,
            num_heads=num_heads,
            input_embed_dim=input_embed_dim,
            num_classes=10,
        )

        net = CustomFTTransformerModel(
            config=model_config,
            columns_features=Config.COLUMNS_FEATURES,
            columns_weights=Config.COLUMNS_WEIGHTS,
            head_dim=head_dim,
            fusion_hidden_dim=fusion_hidden_dim,
            fusion_activation=fusion_activation,
            layer_norm_eps=layer_norm_eps,
            residual_dropout=residual_dropout,
            residual_scale=residual_scale,
        ).to(self.device)

        criterion = DynamicFocalLoss(alpha=None, gamma=focal_gamma, reduction="mean", momentum=focal_momentum)
        optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(beta1, beta2))

        total_steps = self.config.NUM_EPOCHS * len(train_loader)
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=warmup_steps, total_steps=total_steps, lr_decay_factor=lr_decay_factor)

        scaler = torch.amp.GradScaler()

        epoch_results = []
        best_mcc = -float("inf")
        patience_counter = 0

        for epoch in range(self.config.NUM_EPOCHS):
            net.train()
            total_loss = 0

            for features_batch, weights_batch, y_batch in train_loader:
                features_batch, weights_batch, y_batch = features_batch.to(self.device), weights_batch.to(self.device), y_batch.to(self.device)

                with torch.amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
                    outputs = net(features_batch, weights_batch)["logits"]
                    loss = criterion(outputs, y_batch)

                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), gradient_clip_val)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)

            net.eval()
            y_pred = []
            y_true = []
            y_probs = []

            with torch.no_grad():
                for features_batch, weights_batch, y_batch in test_loader:
                    features_batch, weights_batch, y_batch = features_batch.to(self.device), weights_batch.to(self.device), y_batch.to(self.device)
                    outputs = net(features_batch, weights_batch)
                    _, predicted = torch.max(outputs["logits"].data, 1)
                    y_pred.extend(predicted.cpu().numpy())
                    y_true.extend(y_batch.cpu().numpy())
                    y_probs.extend(outputs["probs"].cpu().numpy())

            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            y_probs = np.array(y_probs)

            precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=1)
            accuracy = accuracy_score(y_true, y_pred)
            cohen_kappa = cohen_kappa_score(y_true, y_pred)
            mcc = matthews_corrcoef(y_true, y_pred)
            conf_matrix = confusion_matrix(y_true, y_pred)

            epoch_results.append(
                {
                    "epoch": epoch + 1,
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "cohen_kappa": cohen_kappa,
                    "mcc": mcc,
                    "confusion_matrix": conf_matrix,
                }
            )

            class_mapping = {
                0: 0,
                1: 1,
                2: 2,
                3: 3,
                4: 4,
                5: 0,
                6: 1,
                7: 2,
                8: 3,
                9: 4,
            }
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
                conf_matrix_cls = confusion_matrix(y_true_cls_binary, y_pred_cls_binary)

                if len(np.unique(y_true_cls_binary)) >= 2:
                    auc_cls = roc_auc_score(y_true_cls_binary, y_probs[cls_mask, cls * 2 + 1])
                    ap_cls = average_precision_score(y_true_cls_binary, y_probs[cls_mask, cls * 2 + 1])
                else:
                    auc_cls = np.nan
                    ap_cls = np.nan

                epoch_results[-1][f"cls_{cls}_precision"] = precision_cls
                epoch_results[-1][f"cls_{cls}_recall"] = recall_cls
                epoch_results[-1][f"cls_{cls}_f1"] = f1_cls
                epoch_results[-1][f"cls_{cls}_confusion_matrix"] = conf_matrix_cls
                epoch_results[-1][f"cls_{cls}_auc"] = auc_cls
                epoch_results[-1][f"cls_{cls}_ap"] = ap_cls

            trial.report(accuracy, epoch)

            if mcc > best_mcc:
                best_mcc = mcc
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config.PATIENCE:
                    break

            if trial.should_prune():
                trial.set_user_attr("epoch_results", epoch_results)
                raise optuna.exceptions.TrialPruned()

        trial.set_user_attr("epoch_results", epoch_results)
        return mcc

    def save_results(self, study):
        with open("/home/panxiong/MBT/FTT/results_new8.txt", "w") as f:
            f.write(f"Best trial:\n")
            trial = study.best_trial
            f.write(f"  Value: {trial.value}\n")
            f.write(f"  Params: \n")
            for key, value in trial.params.items():
                f.write(f"    {key}: {value}\n")

            for t in study.trials:
                f.write(f"\nTrial {t.number}:\n")
                f.write(f"  Value: {t.value}\n")
                f.write(f"  Params: {t.params}\n")
                f.write("  Epoch Results:\n")
                for epoch_result in t.user_attrs["epoch_results"]:
                    f.write(
                        f"    Epoch {epoch_result['epoch']}: "
                        f"Accuracy: {epoch_result['accuracy']:.4f}, "
                        f"Precision: {epoch_result['precision']:.4f}, "
                        f"Recall: {epoch_result['recall']:.4f}, "
                        f"F1: {epoch_result['f1']:.4f}, "
                        f"Cohen's Kappa: {epoch_result['cohen_kappa']:.4f}, "
                        f"MCC: {epoch_result['mcc']:.4f}\n"
                    )
                    f.write("    Confusion Matrix:\n")
                    conf_matrix = epoch_result["confusion_matrix"]
                    for row in conf_matrix:
                        row_str = "    " + ", ".join(f"{x:6d}" for x in row)
                        f.write(f"{row_str}\n")

                    for cls in range(5):
                        f.write(f"    Class {cls} \n")
                        f.write(f"      Precision: {epoch_result[f'cls_{cls}_precision']:.4f}\n")
                        f.write(f"      Recall: {epoch_result[f'cls_{cls}_recall']:.4f}\n")
                        f.write(f"      F1: {epoch_result[f'cls_{cls}_f1']:.4f}\n")
                        f.write(f"      AUC: {epoch_result[f'cls_{cls}_auc']:.4f}\n")
                        f.write(f"      AP: {epoch_result[f'cls_{cls}_ap']:.4f}\n")
                        f.write("      Confusion Matrix:\n")
                        conf_matrix_cls = epoch_result[f"cls_{cls}_confusion_matrix"]
                        for row in conf_matrix_cls:
                            row_str = "      " + ", ".join(f"{x:6d}" for x in row)
                            f.write(f"{row_str}\n")


def main():
    config = Config()
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
