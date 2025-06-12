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
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import yaml

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from config import get_config, WEFTTConfig, BaselineConfig, ExperimentConfig
from data_processing import DatasetCreator
from models.we_ftt import create_we_ftt_model, create_ft_transformer_model
from models.baselines import BaselineTrainer
from utils import setup_logging, set_random_seeds, save_model, load_model


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
                from models.components import WarmupCosineSchedule
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
            from utils import TensorDataset
            
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
            return yaml.safe_load(f)
        else:
            return json.load(f)


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
    
    # Prepare data
    logger.info("Preparing data...")
    dataset_creator = DatasetCreator()
    
    if args.data_path:
        train_data, val_data, test_data = dataset_creator.create_training_datasets(args.data_path)
    else:
        # Use default data path
        base_config = get_config('base')
        train_data, val_data, test_data = dataset_creator.create_training_datasets()
    
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
    
    logger.info(f"Training completed. Results saved to {output_dir}")


if __name__ == '__main__':
    main()