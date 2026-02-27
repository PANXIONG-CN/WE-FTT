#!/usr/bin/env python3
"""
Unified training script for WE-FTT project.

This script supports training of multiple models including WE-FTT, 
baseline FT-Transformer, and other baseline models through command-line arguments.

Usage:
    python scripts/train.py --model_name we_ftt --epochs 50 --batch_size 32
    python scripts/train.py --model_name random_forest
    python scripts/train.py --model_name ft_transformer --config_file configs/ft_transformer.yaml
"""

import argparse
import logging
import os
import sys
import json
from dataclasses import asdict
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

try:
    import yaml
except Exception:
    yaml = None

# 添加 WE-FTT 与 src 目录到 Python 路径
weftt_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src_root = os.path.join(weftt_root, "src")
if weftt_root not in sys.path:
    sys.path.insert(0, weftt_root)
if src_root not in sys.path:
    sys.path.insert(0, src_root)

from src.config import get_config, WEFTTConfig, BaselineConfig, ExperimentConfig
from src.models.we_ftt import create_we_ftt_model, create_ft_transformer_model
from src.models.baselines import BaselineTrainer
from src.utils import setup_logging, set_random_seeds, save_model, load_model
from evaluation_protocol.common.splits import load_event_splits, split_df_by_event_id
from evaluation_protocol.common.weighting import add_foldwise_kmeans_weights


logger = logging.getLogger(__name__)


class ModelTrainer:
    """统一模型训练器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_name = config['model_name']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Set random seeds
        set_random_seeds(config.get('random_seed', 42))
        
        # Initialize model
        self.model = self._create_model()
        if self.model is not None:
            self.model.to(self.device)
        
        # Initialize optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Data loaders
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
        logger.info(f"Initialized trainer for {self.model_name}")
    
    def _create_model(self):
        """Create model instance"""
        if self.model_name == 'we_ftt':
            return create_we_ftt_model(
                num_features=len(self.config.get('feature_columns', [])),
                num_classes=self.config.get('num_classes', 2),
                config=self.config.get('model_params', {}),
                use_weight_enhancement=True
            )
        elif self.model_name == 'ft_transformer':
            return create_ft_transformer_model(
                num_features=len(self.config.get('feature_columns', [])),
                num_classes=self.config.get('num_classes', 2),
                config=self.config.get('model_params', {})
            )
        elif self.model_name in ['random_forest', 'catboost', 'tabnet', 'xgboost', 'lightgbm']:
            # For baseline models, we don't create the model here
            # BaselineTrainer will handle model creation and training
            return None
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
    
    def _create_optimizer(self):
        """Create optimizer"""
        if hasattr(self.model, 'parameters'):
            optimizer_params = self.config.get('optimizer_params', {})
            optimizer_type = optimizer_params.get('type', 'adamw')
            
            if optimizer_type == 'adamw':
                return torch.optim.AdamW(
                    self.model.parameters(),
                    lr=optimizer_params.get('lr', 0.001),
                    weight_decay=optimizer_params.get('weight_decay', 0.01)
                )
            elif optimizer_type == 'adam':
                return torch.optim.Adam(
                    self.model.parameters(),
                    lr=optimizer_params.get('lr', 0.001)
                )
            else:
                raise ValueError(f"Unsupported optimizer: {optimizer_type}")
        return None
    
    def _create_scheduler(self):
        """Create learning rate scheduler"""
        if self.optimizer is not None:
            scheduler_params = self.config.get('scheduler_params', {})
            scheduler_type = scheduler_params.get('type', 'cosine')
            
            if scheduler_type == 'cosine':
                from src.models.components import WarmupCosineSchedule
                return WarmupCosineSchedule(
                    self.optimizer,
                    warmup_steps=scheduler_params.get('warmup_steps', 100),
                    t_total=scheduler_params.get('t_total', 1000)
                )
            elif scheduler_type == 'step':
                return torch.optim.lr_scheduler.StepLR(
                    self.optimizer,
                    step_size=scheduler_params.get('step_size', 10),
                    gamma=scheduler_params.get('gamma', 0.1)
                )
        return None
    
    def prepare_data(self, train_data: pd.DataFrame, val_data: pd.DataFrame, test_data: pd.DataFrame):
        """Prepare data for training"""
        if self.model_name in ['we_ftt', 'ft_transformer']:
            # Prepare PyTorch data loaders for deep learning models
            from torch.utils.data import DataLoader
            from src.utils import TensorDataset
            
            feature_cols = self.config['feature_columns']
            weight_cols = self.config.get('weight_columns', [])
            label_col = self.config['label_column']
            
            # Create datasets
            train_dataset = TensorDataset(train_data, feature_cols, weight_cols, label_col)
            val_dataset = TensorDataset(val_data, feature_cols, weight_cols, label_col)
            test_dataset = TensorDataset(test_data, feature_cols, weight_cols, label_col)
            
            # Create data loaders
            batch_size = self.config.get('batch_size', 32)
            self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        else:
            # Prepare data for traditional machine learning models
            feature_cols = self.config['feature_columns']
            weight_cols = self.config.get('weight_columns', [])
            label_col = self.config['label_column']
            
            # Combine features and weights (if any)
            if weight_cols and len(weight_cols) > 0:
                all_feature_cols = feature_cols + weight_cols
            else:
                all_feature_cols = feature_cols
            
            self.X_train = train_data[all_feature_cols].values
            self.y_train = train_data[label_col].values
            self.X_val = val_data[all_feature_cols].values
            self.y_val = val_data[label_col].values
            self.X_test = test_data[all_feature_cols].values
            self.y_test = test_data[label_col].values
    
    def train(self) -> Dict[str, Any]:
        """Train the model"""
        if self.model_name in ['we_ftt', 'ft_transformer']:
            return self._train_deep_model()
        else:
            return self._train_sklearn_model()
    
    def _train_deep_model(self) -> Dict[str, Any]:
        """Train deep learning model"""
        num_epochs = self.config.get('num_epochs', 20)
        patience = self.config.get('patience', 5)
        
        best_val_loss = float('inf')
        patience_counter = 0
        train_history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
        
        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for batch in self.train_loader:
                features, weights, labels = batch
                features, labels = features.to(self.device), labels.to(self.device)
                if weights is not None:
                    weights = weights.to(self.device)
                
                self.optimizer.zero_grad()
                
                if self.model_name == 'we_ftt' and weights is not None:
                    outputs = self.model(features, weights)
                else:
                    outputs = self.model(features)
                
                loss = self.model.compute_loss(outputs, labels, epoch)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
            
            # Validation phase
            val_loss, val_accuracy = self._validate()
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step()
            
            # Record history
            train_history['train_loss'].append(train_loss / len(self.train_loader))
            train_history['val_loss'].append(val_loss)
            train_history['val_accuracy'].append(val_accuracy)
            
            logger.info(f"Epoch {epoch+1}/{num_epochs}: "
                       f"Train Loss: {train_loss/len(self.train_loader):.4f}, "
                       f"Val Loss: {val_loss:.4f}, "
                       f"Val Accuracy: {val_accuracy:.4f}")
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model for testing
        self.model.load_state_dict(torch.load('best_model.pth'))
        test_results = self._test()
        
        return {
            'train_history': train_history,
            'test_results': test_results,
            'best_val_loss': best_val_loss
        }
    
    def _train_sklearn_model(self) -> Dict[str, Any]:
        """Train sklearn/baseline models"""
        logger.info(f"Training {self.model_name} model...")
        
        # Use BaselineTrainer for consistent training
        trainer = BaselineTrainer(self.model_name, self.config)
        results = trainer.train_and_evaluate(
            self.X_train, self.y_train,
            self.X_val, self.y_val, 
            self.X_test, self.y_test
        )
        
        logger.info(f"Validation Accuracy: {results['validation_results']['accuracy']:.4f}")
        logger.info(f"Test Accuracy: {results['test_results']['accuracy']:.4f}")
        
        return results
    
    def _validate(self) -> Tuple[float, float]:
        """Validate the model"""
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                features, weights, labels = batch
                features, labels = features.to(self.device), labels.to(self.device)
                if weights is not None:
                    weights = weights.to(self.device)
                
                if self.model_name == 'we_ftt' and weights is not None:
                    outputs = self.model(features, weights)
                else:
                    outputs = self.model(features)
                
                loss = self.model.compute_loss(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_accuracy = correct / total
        return val_loss / len(self.val_loader), val_accuracy
    
    def _test(self) -> Dict[str, float]:
        """Test the model"""
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in self.test_loader:
                features, weights, labels = batch
                features, labels = features.to(self.device), labels.to(self.device)
                if weights is not None:
                    weights = weights.to(self.device)
                
                if self.model_name == 'we_ftt' and weights is not None:
                    outputs = self.model(features, weights)
                else:
                    outputs = self.model(features)
                
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate various metrics
        from sklearn.metrics import (
            accuracy_score, precision_recall_fscore_support,
            cohen_kappa_score, matthews_corrcoef, roc_auc_score
        )
        
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
        kappa = cohen_kappa_score(all_labels, all_preds)
        mcc = matthews_corrcoef(all_labels, all_preds)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'cohen_kappa': kappa,
            'matthews_corrcoef': mcc
        }


def load_config_file(config_path: str) -> Dict[str, Any]:
    """Load configuration file"""
    with open(config_path, 'r') as f:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            if yaml is None:
                raise ImportError("缺少依赖 pyyaml，无法读取 YAML 配置文件。")
            return yaml.safe_load(f)
        else:
            return json.load(f)


def _load_training_dataframe(data_path: str) -> pd.DataFrame:
    p = str(data_path)
    if p.endswith(".parquet"):
        return pd.read_parquet(p)
    if p.endswith(".csv"):
        return pd.read_csv(p)
    raise ValueError(f"Unsupported file format: {data_path}")


def _require_columns(df: pd.DataFrame, columns: list[str], *, context: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{context} 缺少列: {missing}")


def _is_binary_series(series: pd.Series) -> bool:
    if series.empty:
        return False
    if not pd.api.types.is_numeric_dtype(series):
        return False
    vals = set(pd.Series(series).dropna().astype(int).tolist())
    return vals.issubset({0, 1})


def _split_dataframe(
    df: pd.DataFrame,
    *,
    label_col: str,
    splits_json: Optional[str],
    random_seed: int,
    test_size: float,
    val_size: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    if splits_json:
        _require_columns(df, ["event_id"], context="事件级切分")
        splits = load_event_splits(splits_json)
        train_df, val_df, test_df = split_df_by_event_id(df, splits)
        if train_df.empty or val_df.empty or test_df.empty:
            raise ValueError("事件级切分后 train/val/test 至少有一个为空。")
        meta = {
            "split_mode": "event_grouped",
            "splits_json": str(splits_json),
            "n_events": {
                "train": int(len(splits.train_event_ids)),
                "val": int(len(splits.val_event_ids)),
                "test": int(len(splits.test_event_ids)),
            },
        }
        return train_df.copy(), val_df.copy(), test_df.copy(), meta

    if float(test_size) <= 0 or float(val_size) <= 0 or float(test_size + val_size) >= 1:
        raise ValueError("test_size 与 val_size 必须 >0 且 test_size+val_size < 1。")

    y = df[label_col]
    stratify = y if y.nunique(dropna=False) > 1 else None
    temp_df, test_df = train_test_split(
        df,
        test_size=float(test_size),
        random_state=int(random_seed),
        stratify=stratify,
    )
    val_ratio_adjusted = float(val_size) / (1.0 - float(test_size))
    y_temp = temp_df[label_col]
    stratify_temp = y_temp if y_temp.nunique(dropna=False) > 1 else None
    train_df, val_df = train_test_split(
        temp_df,
        test_size=float(val_ratio_adjusted),
        random_state=int(random_seed),
        stratify=stratify_temp,
    )
    meta = {
        "split_mode": "random_stratified",
        "test_size": float(test_size),
        "val_size": float(val_size),
    }
    return train_df.copy(), val_df.copy(), test_df.copy(), meta


def _encode_binary_labels(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    label_col: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    y_train = train_df[label_col]
    if pd.api.types.is_numeric_dtype(y_train):
        train_vals = set(pd.Series(y_train).dropna().astype(int).tolist())
        all_vals = set(pd.concat([train_df[label_col], val_df[label_col], test_df[label_col]], axis=0).dropna().astype(int).tolist())
        if all_vals.issubset({0, 1}):
            for d in (train_df, val_df, test_df):
                d[label_col] = d[label_col].astype(int)
            return train_df, val_df, test_df, {"label_mode": "already_binary", "classes": [0, 1]}

    encoder = LabelEncoder()
    encoder.fit(train_df[label_col].astype(str))
    train_classes = set(encoder.classes_.tolist())

    def _transform(df_in: pd.DataFrame) -> pd.DataFrame:
        unknown = set(df_in[label_col].astype(str).unique().tolist()) - train_classes
        if unknown:
            raise ValueError(f"{label_col} 在训练集未出现的类别: {sorted(list(unknown))}")
        df_out = df_in.copy()
        df_out[label_col] = encoder.transform(df_out[label_col].astype(str)).astype(int)
        return df_out

    train_df = _transform(train_df)
    val_df = _transform(val_df)
    test_df = _transform(test_df)

    all_vals = set(pd.concat([train_df[label_col], val_df[label_col], test_df[label_col]], axis=0).dropna().astype(int).tolist())
    if not all_vals.issubset({0, 1}):
        raise ValueError(f"当前仅支持二分类标签（0/1），实际编码类别: {sorted(list(all_vals))}")
    return train_df, val_df, test_df, {"label_mode": "encoded_from_strings", "classes": encoder.classes_.tolist()}


def _fit_transform_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    feature_columns: list[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    scaler = StandardScaler()
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    train_df[feature_columns] = scaler.fit_transform(train_df[feature_columns].astype(float))
    val_df[feature_columns] = scaler.transform(val_df[feature_columns].astype(float))
    test_df[feature_columns] = scaler.transform(test_df[feature_columns].astype(float))
    return train_df, val_df, test_df, {
        "scaler_mean": scaler.mean_.astype(float).tolist(),
        "scaler_scale": scaler.scale_.astype(float).tolist(),
    }


def _apply_foldwise_weights_if_needed(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    feature_columns: list[str],
    weight_columns: list[str],
    label_col: str,
    enable_foldwise_weights: bool,
    n_clusters: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    if not weight_columns or not enable_foldwise_weights:
        return train_df, val_df, test_df, {
            "foldwise_weights_enabled": bool(enable_foldwise_weights),
            "artifacts": [],
        }

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    for d in (train_df, val_df, test_df):
        existing_w = [c for c in weight_columns if c in d.columns]
        if existing_w:
            d.drop(columns=existing_w, inplace=True)

    train_df, [val_df, test_df], artifacts = add_foldwise_kmeans_weights(
        train_df=train_df,
        other_dfs=[val_df, test_df],
        feature_columns=feature_columns,
        n_clusters=int(n_clusters),
        seed=int(seed),
        flag_col=label_col,
    )
    return train_df, val_df, test_df, {
        "foldwise_weights_enabled": True,
        "n_clusters": int(n_clusters),
        "artifacts": [asdict(a) for a in artifacts],
    }


def prepare_training_datasets(config: Dict[str, Any], args) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    feature_cols = list(config["feature_columns"])
    weight_cols = list(config.get("weight_columns", []) or [])
    label_col = str(config["label_column"])

    data_path = args.data_path or str(get_config("base").TRAINING_DATASET)
    df = _load_training_dataframe(data_path)
    _require_columns(df, feature_cols, context="训练数据")

    if label_col not in df.columns:
        if "flag" in df.columns:
            logger.warning("标签列 %s 不存在，自动回退到二分类列 flag。", label_col)
            label_col = "flag"
            config["label_column"] = "flag"
        else:
            raise ValueError(f"训练数据缺少标签列: {label_col}")

    if not _is_binary_series(df[label_col]) and "flag" in df.columns and _is_binary_series(df["flag"]):
        logger.warning(
            "标签列 %s 非二分类，自动回退到二分类列 flag（原标签唯一值=%d）。",
            label_col,
            int(df[label_col].nunique(dropna=True)),
        )
        label_col = "flag"
        config["label_column"] = "flag"

    if int(args.max_rows) > 0 and len(df) > int(args.max_rows):
        y = df[label_col]
        stratify = y if y.nunique(dropna=False) > 1 else None
        _, df = train_test_split(
            df,
            test_size=float(args.max_rows) / float(len(df)),
            random_state=int(args.random_seed),
            stratify=stratify,
        )
        df = df.copy()

    df = df.drop_duplicates().reset_index(drop=True)

    train_df, val_df, test_df, split_meta = _split_dataframe(
        df,
        label_col=label_col,
        splits_json=args.splits_json,
        random_seed=int(args.random_seed),
        test_size=float(args.test_size),
        val_size=float(args.val_size),
    )

    train_df, val_df, test_df, label_meta = _encode_binary_labels(
        train_df,
        val_df,
        test_df,
        label_col=label_col,
    )
    train_df, val_df, test_df, scaler_meta = _fit_transform_features(
        train_df,
        val_df,
        test_df,
        feature_columns=feature_cols,
    )

    train_df, val_df, test_df, weighting_meta = _apply_foldwise_weights_if_needed(
        train_df,
        val_df,
        test_df,
        feature_columns=feature_cols,
        weight_columns=weight_cols,
        label_col=label_col,
        enable_foldwise_weights=not bool(args.disable_foldwise_weights),
        n_clusters=int(args.n_clusters),
        seed=int(args.random_seed),
    )

    meta = {
        "data_path": str(data_path),
        "n_rows": {
            "total": int(len(df)),
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "test": int(len(test_df)),
        },
        "split": split_meta,
        "label": label_meta,
        "scaler": scaler_meta,
        "weighting": weighting_meta,
    }
    return train_df, val_df, test_df, meta


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train models for WE-FTT project')
    
    parser.add_argument('--model_name', type=str, required=True,
                       choices=['we_ftt', 'ft_transformer', 'random_forest', 'catboost', 'tabnet', 'xgboost', 'lightgbm'],
                       help='Model to train')
    
    parser.add_argument('--config_file', type=str, default=None,
                       help='Path to configuration file')
    
    parser.add_argument('--data_path', type=str, default=None,
                       help='Path to training data')

    parser.add_argument('--splits_json', type=str, default=None,
                       help='事件级切分清单JSON路径（可选，提供后启用事件级切分）')

    parser.add_argument('--test_size', type=float, default=0.2,
                       help='随机切分测试集占比（仅在未提供 --splits_json 时生效）')

    parser.add_argument('--val_size', type=float, default=0.2,
                       help='随机切分验证集占比（仅在未提供 --splits_json 时生效）')

    parser.add_argument('--disable_foldwise_weights', action='store_true',
                       help='关闭训练折内权重重算（默认开启）')

    parser.add_argument('--n_clusters', type=int, default=5,
                       help='fold-wise KMeans 聚类数（默认5）')

    parser.add_argument('--max_rows', type=int, default=0,
                       help='仅用于调试：限制读取后的最大样本数（0表示不限制）')

    parser.add_argument('--output_dir', type=str, default='results',
                       help='Output directory for results')
    
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs')
    
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    
    parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID')
    
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()
    
    # Setup logging
    setup_logging()
    
    # Setup GPU
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        logger.info(f"Using GPU {args.gpu}")
    else:
        logger.info("Using CPU")
    
    # Load configuration
    if args.config_file:
        config = load_config_file(args.config_file)
    else:
        # Use default configuration
        if args.model_name == 'we_ftt':
            base_config = WEFTTConfig()
            config = {
                'model_name': args.model_name,
                'model_params': base_config.BEST_PARAMS,
                'feature_columns': base_config.COLUMNS_FEATURES,
                'weight_columns': base_config.COLUMNS_WEIGHTS,
                'label_column': base_config.LABEL_COLUMN
            }
        elif args.model_name == 'ft_transformer':
            base_config = WEFTTConfig()
            config = {
                'model_name': args.model_name,
                'model_params': base_config.BEST_PARAMS,
                'feature_columns': base_config.COLUMNS_FEATURES,
                'label_column': base_config.LABEL_COLUMN
            }
        else:
            baseline_config = BaselineConfig()
            config = {
                'model_name': args.model_name,
                'model_params': getattr(baseline_config, f"{args.model_name.upper()}_PARAMS", {}),
                'feature_columns': baseline_config.COLUMNS_FEATURES,
                'weight_columns': baseline_config.COLUMNS_WEIGHTS,
                'label_column': baseline_config.LABEL_COLUMN
            }
    
    # Override with command line arguments
    config.update({
        'num_epochs': args.epochs,
        'batch_size': args.batch_size,
        'random_seed': args.random_seed
    })
    
    if 'optimizer_params' not in config:
        config['optimizer_params'] = {'lr': args.learning_rate}
    else:
        config['optimizer_params']['lr'] = args.learning_rate
    
    # Prepare data（主训练链：切分后拟合，fold-wise 防泄漏）
    logger.info("Preparing data with leakage-safe pipeline...")
    train_data, val_data, test_data, data_meta = prepare_training_datasets(config, args)
    logger.info(
        "Data prepared. total=%d, train=%d, val=%d, test=%d, split_mode=%s, foldwise_weights=%s",
        data_meta["n_rows"]["total"],
        data_meta["n_rows"]["train"],
        data_meta["n_rows"]["val"],
        data_meta["n_rows"]["test"],
        data_meta["split"]["split_mode"],
        data_meta["weighting"]["foldwise_weights_enabled"],
    )
    
    # Initialize trainer
    trainer = ModelTrainer(config)
    trainer.prepare_data(train_data, val_data, test_data)
    
    # Train model
    logger.info(f"Starting training for {args.model_name}...")
    results = trainer.train()
    
    # Save results
    output_dir = Path(args.output_dir) / args.model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save configuration
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    with open(output_dir / 'data_pipeline_meta.json', 'w') as f:
        json.dump(data_meta, f, indent=2, default=str)
    
    logger.info(f"Training completed. Results saved to {output_dir}")


if __name__ == '__main__':
    main()
