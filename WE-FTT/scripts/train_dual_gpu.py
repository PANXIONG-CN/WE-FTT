#!/usr/bin/env python3
"""
双GPU分布式训练脚本 - 针对两个RTX 3090
使用PyTorch DistributedDataParallel进行训练

使用方法:
    CUDA_VISIBLE_DEVICES=0,1 python scripts/train_dual_gpu.py --model_name we_ftt --epochs 50
"""

import argparse
import logging
import os
import sys
import json
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple
from datetime import datetime

# 添加项目根目录到Python路径
project_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, project_root)

from src.config import get_config, WEFTTConfig, BaselineConfig
from src.data_processing import DatasetCreator
from src.models.we_ftt import create_we_ftt_model, create_ft_transformer_model
from src.utils import setup_logging, set_random_seeds


logger = logging.getLogger(__name__)


def setup_distributed(rank, world_size):
    """初始化分布式训练环境"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # 初始化进程组
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    if rank == 0:
        logger.info(f"Initialized distributed training with {world_size} GPUs")


def cleanup_distributed():
    """清理分布式训练环境"""
    dist.destroy_process_group()


class DistributedTrainer:
    """分布式训练器"""

    def __init__(self, rank, world_size, config: Dict[str, Any]):
        self.rank = rank
        self.world_size = world_size
        self.config = config
        self.device = torch.device(f'cuda:{rank}')

        # 只在rank 0上打印日志
        self.is_main_process = (rank == 0)

        # 设置随机种子
        set_random_seeds(config.get('random_seed', 42) + rank)

        # 创建模型
        self.model = self._create_model()
        self.model.to(self.device)

        # 包装为DDP模型
        self.model = DDP(self.model, device_ids=[rank], find_unused_parameters=False)

        # 创建优化器和调度器
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()

        # 混合精度训练
        self.scaler = torch.cuda.amp.GradScaler()

        # 最佳模型保存
        self.best_val_loss = float('inf')
        self.best_model_path = Path(config.get('output_dir', 'results')) / 'best_model.pth'
        self.best_model_path.parent.mkdir(parents=True, exist_ok=True)

        if self.is_main_process:
            logger.info(f"Initialized trainer on rank {rank}")
            logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")

    def _create_model(self):
        """创建模型"""
        model_name = self.config['model_name']

        if model_name == 'we_ftt':
            return create_we_ftt_model(
                num_features=len(self.config.get('feature_columns', [])),
                num_classes=self.config.get('num_classes', 2),
                config=self.config.get('model_params', {}),
                use_weight_enhancement=True
            )
        elif model_name == 'ft_transformer':
            return create_ft_transformer_model(
                num_features=len(self.config.get('feature_columns', [])),
                num_classes=self.config.get('num_classes', 2),
                config=self.config.get('model_params', {})
            )
        else:
            raise ValueError(f"Unsupported model: {model_name}")

    def _create_optimizer(self):
        """创建优化器"""
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

    def _create_scheduler(self):
        """创建学习率调度器"""
        scheduler_params = self.config.get('scheduler_params', {})
        scheduler_type = scheduler_params.get('type', 'cosine')

        if scheduler_type == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.get('num_epochs', 50),
                eta_min=1e-6
            )
        elif scheduler_type == 'step':
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_params.get('step_size', 10),
                gamma=scheduler_params.get('gamma', 0.1)
            )
        else:
            return None

    def prepare_data_loaders(self, train_data, val_data, test_data):
        """准备分布式数据加载器"""
        from torch.utils.data import DataLoader, TensorDataset

        feature_cols = self.config['feature_columns']
        weight_cols = self.config.get('weight_columns', [])
        label_col = self.config['label_column']

        # 准备张量
        X_train = torch.FloatTensor(train_data[feature_cols].values)
        y_train = torch.LongTensor(train_data[label_col].values)

        X_val = torch.FloatTensor(val_data[feature_cols].values)
        y_val = torch.LongTensor(val_data[label_col].values)

        X_test = torch.FloatTensor(test_data[feature_cols].values)
        y_test = torch.LongTensor(test_data[label_col].values)

        # 权重数据（如果有）
        if weight_cols and len(weight_cols) > 0:
            W_train = torch.FloatTensor(train_data[weight_cols].values)
            W_val = torch.FloatTensor(val_data[weight_cols].values)
            W_test = torch.FloatTensor(test_data[weight_cols].values)

            train_dataset = TensorDataset(X_train, W_train, y_train)
            val_dataset = TensorDataset(X_val, W_val, y_val)
            test_dataset = TensorDataset(X_test, W_test, y_test)
        else:
            # 创建空权重张量
            W_train = torch.zeros(X_train.size(0), len(feature_cols))
            W_val = torch.zeros(X_val.size(0), len(feature_cols))
            W_test = torch.zeros(X_test.size(0), len(feature_cols))

            train_dataset = TensorDataset(X_train, W_train, y_train)
            val_dataset = TensorDataset(X_val, W_val, y_val)
            test_dataset = TensorDataset(X_test, W_test, y_test)

        # 创建分布式采样器
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True
        )

        # 数据加载器
        batch_size = self.config.get('batch_size', 256)  # 每个GPU的批量大小

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=4,
            pin_memory=True
        )

        # 验证和测试不需要分布式采样
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        self.test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        if self.is_main_process:
            logger.info(f"Data loaders prepared: Train batches={len(self.train_loader)}, "
                       f"Val batches={len(self.val_loader)}, Test batches={len(self.test_loader)}")

    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        self.train_loader.sampler.set_epoch(epoch)  # 重要：确保每个epoch的数据顺序不同

        total_loss = 0.0
        num_batches = 0

        for batch_idx, (features, weights, labels) in enumerate(self.train_loader):
            features = features.to(self.device)
            weights = weights.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            # 混合精度训练
            with torch.cuda.amp.autocast():
                if self.config['model_name'] == 'we_ftt' and weights.sum() != 0:
                    outputs = self.model(features, weights)
                else:
                    outputs = self.model(features)

                loss = self.model.module.compute_loss(outputs, labels, epoch)

            # 反向传播
            self.scaler.scale(loss).backward()

            # 梯度裁剪
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            num_batches += 1

            # 定期打印进度
            if self.is_main_process and batch_idx % 100 == 0:
                logger.info(f"Epoch {epoch}, Batch {batch_idx}/{len(self.train_loader)}, "
                           f"Loss: {loss.item():.4f}")

        avg_loss = total_loss / num_batches
        return avg_loss

    @torch.no_grad()
    def validate(self):
        """验证模型"""
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        for features, weights, labels in self.val_loader:
            features = features.to(self.device)
            weights = weights.to(self.device)
            labels = labels.to(self.device)

            with torch.cuda.amp.autocast():
                if self.config['model_name'] == 'we_ftt' and weights.sum() != 0:
                    outputs = self.model(features, weights)
                else:
                    outputs = self.model(features)

                loss = self.model.module.compute_loss(outputs, labels)

            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct / total

        return avg_loss, accuracy

    @torch.no_grad()
    def test(self):
        """测试模型"""
        self.model.eval()

        all_preds = []
        all_labels = []
        all_probs = []

        for features, weights, labels in self.test_loader:
            features = features.to(self.device)
            weights = weights.to(self.device)
            labels = labels.to(self.device)

            with torch.cuda.amp.autocast():
                if self.config['model_name'] == 'we_ftt' and weights.sum() != 0:
                    outputs = self.model(features, weights)
                else:
                    outputs = self.model(features)

            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

        # 计算指标
        from sklearn.metrics import (
            accuracy_score, precision_recall_fscore_support,
            cohen_kappa_score, matthews_corrcoef, roc_auc_score
        )

        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted'
        )
        kappa = cohen_kappa_score(all_labels, all_preds)
        mcc = matthews_corrcoef(all_labels, all_preds)

        all_probs = np.array(all_probs)
        if all_probs.shape[1] == 2:
            roc_auc = roc_auc_score(all_labels, all_probs[:, 1])
        else:
            roc_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='weighted')

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'cohen_kappa': kappa,
            'matthews_corrcoef': mcc,
            'roc_auc': roc_auc
        }

    def train(self):
        """完整训练流程"""
        num_epochs = self.config.get('num_epochs', 50)
        patience = self.config.get('patience', 10)
        patience_counter = 0

        train_history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'learning_rate': []
        }

        if self.is_main_process:
            logger.info("=" * 60)
            logger.info("开始训练")
            logger.info(f"总epochs: {num_epochs}")
            logger.info(f"早停patience: {patience}")
            logger.info(f"使用GPU数量: {self.world_size}")
            logger.info("=" * 60)

        for epoch in range(num_epochs):
            epoch_start_time = datetime.now()

            # 训练
            train_loss = self.train_epoch(epoch)

            # 验证
            val_loss, val_accuracy = self.validate()

            # 更新学习率
            if self.scheduler:
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
            else:
                current_lr = self.config['optimizer_params']['lr']

            # 记录历史
            train_history['train_loss'].append(train_loss)
            train_history['val_loss'].append(val_loss)
            train_history['val_accuracy'].append(val_accuracy)
            train_history['learning_rate'].append(current_lr)

            epoch_time = (datetime.now() - epoch_start_time).total_seconds()

            if self.is_main_process:
                logger.info(f"Epoch {epoch+1}/{num_epochs} - "
                           f"Train Loss: {train_loss:.4f}, "
                           f"Val Loss: {val_loss:.4f}, "
                           f"Val Acc: {val_accuracy:.4f}, "
                           f"LR: {current_lr:.6f}, "
                           f"Time: {epoch_time:.1f}s")

            # 保存最佳模型（只在主进程）
            if self.is_main_process and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                patience_counter = 0

                # 保存模型
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.module.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_accuracy': val_accuracy,
                    'config': self.config
                }, self.best_model_path)

                logger.info(f"✓ 保存最佳模型 (Val Loss: {val_loss:.4f})")
            else:
                patience_counter += 1

            # 早停检查
            if patience_counter >= patience:
                if self.is_main_process:
                    logger.info(f"早停触发 at epoch {epoch+1}")
                break

        # 加载最佳模型进行测试
        if self.is_main_process:
            logger.info("\n" + "=" * 60)
            logger.info("训练完成，加载最佳模型进行测试")
            logger.info("=" * 60)

            checkpoint = torch.load(self.best_model_path)
            self.model.module.load_state_dict(checkpoint['model_state_dict'])

        # 同步所有进程
        dist.barrier()

        # 测试
        test_results = self.test()

        if self.is_main_process:
            logger.info("\n" + "=" * 60)
            logger.info("测试结果:")
            for metric, value in test_results.items():
                logger.info(f"{metric}: {value:.4f}")
            logger.info("=" * 60)

        return {
            'train_history': train_history,
            'test_results': test_results,
            'best_val_loss': self.best_val_loss
        }


def main_worker(rank, world_size, args):
    """每个GPU上的工作进程"""
    # 设置分布式
    setup_distributed(rank, world_size)

    # 只在主进程设置日志
    if rank == 0:
        setup_logging()

    # 加载配置
    if args.model_name == 'we_ftt':
        base_config = WEFTTConfig()
        config = {
            'model_name': args.model_name,
            'model_params': base_config.BEST_PARAMS,
            'feature_columns': base_config.COLUMNS_FEATURES,
            'weight_columns': base_config.COLUMNS_WEIGHTS,
            'label_column': base_config.LABEL_COLUMN,
            'num_classes': 2
        }
    elif args.model_name == 'ft_transformer':
        base_config = WEFTTConfig()
        config = {
            'model_name': args.model_name,
            'model_params': base_config.BEST_PARAMS,
            'feature_columns': base_config.COLUMNS_FEATURES,
            'label_column': base_config.LABEL_COLUMN,
            'num_classes': 2
        }
    else:
        raise ValueError(f"不支持的模型: {args.model_name}")

    # 覆盖命令行参数
    config.update({
        'num_epochs': args.epochs,
        'batch_size': args.batch_size,  # 每个GPU的批量大小
        'random_seed': args.random_seed,
        'patience': args.patience,
        'output_dir': args.output_dir,
        'optimizer_params': {
            'type': 'adamw',
            'lr': args.learning_rate,
            'weight_decay': args.weight_decay
        },
        'scheduler_params': {
            'type': 'cosine'
        }
    })

    # 加载数据（只在主进程加载一次）
    if rank == 0:
        logger.info("加载数据...")

    dataset_creator = DatasetCreator()
    if args.data_path:
        train_data, val_data, test_data = dataset_creator.create_training_datasets(args.data_path)
    else:
        train_data, val_data, test_data = dataset_creator.create_training_datasets()

    if rank == 0:
        logger.info(f"数据加载完成: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")

    # 创建训练器
    trainer = DistributedTrainer(rank, world_size, config)
    trainer.prepare_data_loaders(train_data, val_data, test_data)

    # 训练
    results = trainer.train()

    # 保存结果（只在主进程）
    if rank == 0:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / 'results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)

        with open(output_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)

        logger.info(f"\n结果已保存到: {output_dir}")
        logger.info(f"最佳模型保存在: {trainer.best_model_path}")

    # 清理
    cleanup_distributed()


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='双GPU分布式训练WE-FTT模型')

    parser.add_argument('--model_name', type=str, default='we_ftt',
                       choices=['we_ftt', 'ft_transformer'],
                       help='要训练的模型')

    parser.add_argument('--data_path', type=str, default=None,
                       help='训练数据路径')

    parser.add_argument('--output_dir', type=str, default='results/dual_gpu_training',
                       help='输出目录')

    parser.add_argument('--epochs', type=int, default=50,
                       help='训练轮数')

    parser.add_argument('--batch_size', type=int, default=256,
                       help='每个GPU的批量大小')

    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='学习率')

    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='权重衰减')

    parser.add_argument('--patience', type=int, default=10,
                       help='早停的patience')

    parser.add_argument('--random_seed', type=int, default=42,
                       help='随机种子')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # 使用2个GPU
    world_size = 2

    # 启动多进程训练
    mp.spawn(
        main_worker,
        args=(world_size, args),
        nprocs=world_size,
        join=True
    )
