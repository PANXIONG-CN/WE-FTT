#!/usr/bin/env python3
"""
流式训练脚本 - 不加载全部数据到内存
使用批量读取parquet文件，节省内存
"""

import argparse
import logging
import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from typing import Iterator, Tuple
import pyarrow.parquet as pq

# 添加项目根目录到路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.config import WEFTTConfig
from src.models.we_ftt import create_we_ftt_model
from src.utils import setup_logging, set_random_seeds
from torch.utils.data import IterableDataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


logger = logging.getLogger(__name__)


class StreamingParquetDataset(IterableDataset):
    """
    流式Parquet数据集 - 按批次读取数据，不占用大量内存
    """
    
    def __init__(self, parquet_file: str, feature_cols: list, weight_cols: list, 
                 label_col: str, batch_size: int = 10000, shuffle: bool = True):
        self.parquet_file = parquet_file
        self.feature_cols = feature_cols
        self.weight_cols = weight_cols
        self.label_col = label_col
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # 获取数据集总行数
        parquet_file_obj = pq.ParquetFile(parquet_file)
        self.total_rows = parquet_file_obj.metadata.num_rows
        self.num_row_groups = parquet_file_obj.num_row_groups
        
        logger.info(f"StreamingDataset initialized: {self.total_rows:,} rows, "
                   f"{self.num_row_groups} row groups")
    
    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """迭代数据批次"""
        parquet_file = pq.ParquetFile(self.parquet_file)
        
        # 按row group读取
        for batch in parquet_file.iter_batches(batch_size=self.batch_size):
            df = batch.to_pandas()
            
            # 提取特征、权重和标签
            features = df[self.feature_cols].values.astype(np.float32)
            labels = df[self.label_col].values.astype(np.int64)
            
            if self.weight_cols and len(self.weight_cols) > 0:
                weights = df[self.weight_cols].values.astype(np.float32)
            else:
                weights = np.ones((len(df), len(self.feature_cols)), dtype=np.float32)
            
            # 转换为tensor
            features_tensor = torch.from_numpy(features)
            weights_tensor = torch.from_numpy(weights)
            labels_tensor = torch.from_numpy(labels)
            
            # 按单个样本返回
            for i in range(len(df)):
                yield features_tensor[i], weights_tensor[i], labels_tensor[i]


def collate_fn(batch):
    """自定义collate函数"""
    features, weights, labels = zip(*batch)
    return (
        torch.stack(features),
        torch.stack(weights),
        torch.stack(labels)
    )


class StreamingTrainer:
    """流式训练器"""
    
    def __init__(self, config: dict, data_file: str, output_dir: Path):
        self.config = config
        self.data_file = data_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        set_random_seeds(config.get('random_seed', 42))
        
        # 创建模型
        self.model = create_we_ftt_model(
            num_features=len(config['feature_columns']),
            num_classes=config.get('num_classes', 2),
            config=config.get('model_params', {}),
            use_weight_enhancement=True
        )
        self.model.to(self.device)
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.get('learning_rate', 0.001),
            weight_decay=0.01
        )
        
        logger.info(f"StreamingTrainer initialized on {self.device}")
    
    def create_dataloader(self, shuffle: bool = True) -> DataLoader:
        """创建流式DataLoader"""
        dataset = StreamingParquetDataset(
            parquet_file=self.data_file,
            feature_cols=self.config['feature_columns'],
            weight_cols=self.config.get('weight_columns', []),
            label_col=self.config['label_column'],
            batch_size=self.config.get('stream_batch_size', 10000),
            shuffle=shuffle
        )
        
        return DataLoader(
            dataset,
            batch_size=self.config.get('batch_size', 128),
            collate_fn=collate_fn,
            num_workers=0  # 使用主进程读取
        )
    
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (features, weights, labels) in enumerate(dataloader):
            features = features.to(self.device)
            weights = weights.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(features, weights)
            loss = self.model.compute_loss(outputs, labels, epoch)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if (batch_idx + 1) % 100 == 0:
                logger.info(f"  Batch {batch_idx+1}: Loss={loss.item():.4f}")
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    def validate(self, dataloader: DataLoader) -> Tuple[float, float]:
        """验证"""
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for features, weights, labels in dataloader:
                features = features.to(self.device)
                weights = weights.to(self.device)
                
                outputs = self.model(features, weights)
                _, predicted = torch.max(outputs, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        accuracy = accuracy_score(all_labels, all_preds)
        return 0.0, accuracy  # 返回损失和准确率
    
    def train(self, num_epochs: int = 20):
        """完整训练流程"""
        logger.info(f"Starting streaming training for {num_epochs} epochs...")
        
        # 创建dataloader（每个epoch重新创建以shuffle数据）
        best_accuracy = 0.0
        train_history = {'train_loss': [], 'val_accuracy': []}
        
        for epoch in range(num_epochs):
            logger.info(f"\nEpoch {epoch+1}/{num_epochs}")
            logger.info("="*60)
            
            # 训练
            train_loader = self.create_dataloader(shuffle=True)
            train_loss = self.train_epoch(train_loader, epoch)
            train_history['train_loss'].append(train_loss)
            
            logger.info(f"Train Loss: {train_loss:.4f}")
            
            # 每5个epoch验证一次
            if (epoch + 1) % 5 == 0:
                logger.info("Validating...")
                val_loader = self.create_dataloader(shuffle=False)
                _, val_accuracy = self.validate(val_loader)
                train_history['val_accuracy'].append(val_accuracy)
                logger.info(f"Validation Accuracy: {val_accuracy:.4f}")
                
                # 保存最佳模型
                if val_accuracy > best_accuracy:
                    best_accuracy = val_accuracy
                    checkpoint = {
                        'epoch': epoch + 1,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'val_accuracy': val_accuracy,
                        'config': self.config
                    }
                    checkpoint_path = self.output_dir / 'best_model.pth'
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"✓ Saved best model (accuracy: {val_accuracy:.4f})")
        
        # 保存训练历史
        with open(self.output_dir / 'results.json', 'w') as f:
            json.dump({
                'train_history': train_history,
                'best_accuracy': best_accuracy
            }, f, indent=2)
        
        logger.info("\n" + "="*60)
        logger.info("Training completed!")
        logger.info(f"Best validation accuracy: {best_accuracy:.4f}")
        logger.info(f"Results saved to: {self.output_dir}")
        logger.info("="*60)


def main():
    parser = argparse.ArgumentParser(description='Streaming training for WE-FTT')
    parser.add_argument('--data_path', type=str, required=True, help='Path to parquet file')
    parser.add_argument('--output_dir', type=str, default='results/we_ftt_streaming',
                       help='Output directory')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--stream_batch_size', type=int, default=10000,
                       help='Streaming batch size (rows to read at once)')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging()
    
    # 设置GPU
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        logger.info(f"Using GPU {args.gpu}")
    else:
        logger.info("Using CPU")
    
    # 配置
    base_config = WEFTTConfig()
    config = {
        'model_name': 'we_ftt',
        'model_params': base_config.BEST_PARAMS,
        'feature_columns': base_config.COLUMNS_FEATURES,
        'weight_columns': base_config.COLUMNS_WEIGHTS,
        'label_column': base_config.LABEL_COLUMN,
        'num_classes': 2,
        'batch_size': args.batch_size,
        'stream_batch_size': args.stream_batch_size,
        'learning_rate': args.learning_rate,
        'random_seed': 42
    }
    
    # 创建训练器
    trainer = StreamingTrainer(config, args.data_path, args.output_dir)
    
    # 训练
    trainer.train(num_epochs=args.epochs)


if __name__ == '__main__':
    main()
