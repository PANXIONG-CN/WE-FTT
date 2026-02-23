#!/usr/bin/env python3
"""
Ablation study script for WE-FTT project.

This script runs comprehensive ablation studies to analyze the contribution
of different components in the WE-FTT model.

Usage:
    python scripts/run_ablation.py --output_dir results/ablation_study
    python scripts/run_ablation.py --config_file configs/ablation.yaml --parallel
"""

import argparse
import logging
import os
import sys
import json
import torch
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List
import yaml
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from config import get_config, WEFTTConfig, ExperimentConfig
from data_processing import DatasetCreator
from models.we_ftt import create_we_ftt_model
from utils import setup_logging, set_random_seeds


logger = logging.getLogger(__name__)


class AblationStudy:
    """消融实验管理器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.base_config = WEFTTConfig()
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置随机种子
        set_random_seeds(config.get('random_seed', 42))
        
        # 设备配置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Initialized ablation study")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Using device: {self.device}")
    
    def define_ablation_experiments(self) -> Dict[str, Dict[str, Any]]:
        """定义消融实验配置"""
        
        base_params = self.base_config.BEST_PARAMS.copy()
        
        experiments = {
            # 基线模型（完整的WE-FTT）
            'full_model': {
                'description': 'Complete WE-FTT model with all components',
                'params': base_params.copy(),
                'modifications': {}
            },
            
            # 移除权重增强
            'no_weight_enhancement': {
                'description': 'WE-FTT without weight enhancement',
                'params': base_params.copy(),
                'modifications': {'use_weight_enhancement': False}
            },
            
            # 移除位置编码
            'no_pos_encoding': {
                'description': 'WE-FTT without positional encoding',
                'params': base_params.copy(),
                'modifications': {'use_positional_encoding': False}
            },
            
            # 可学习位置编码
            'learnable_pos_encoding': {
                'description': 'WE-FTT with learnable positional encoding',
                'params': base_params.copy(),
                'modifications': {'learnable_pos_enc': True}
            },
            
            # 移除残差连接
            'no_residual': {
                'description': 'WE-FTT without residual connections',
                'params': base_params.copy(),
                'modifications': {'prenormalization': False, 'residual_dropout': 1.0}
            },
            
            # 单头注意力
            'single_head_attention': {
                'description': 'WE-FTT with single-head attention',
                'params': base_params.copy(),
                'modifications': {'n_heads': 1}
            },
            
            # 不同激活函数
            'gelu_activation': {
                'description': 'WE-FTT with GELU activation',
                'params': base_params.copy(),
                'modifications': {'activation': 'gelu', 'head_activation': 'gelu'}
            },
            
            'swish_activation': {
                'description': 'WE-FTT with Swish activation',
                'params': base_params.copy(),
                'modifications': {'activation': 'swish', 'head_activation': 'swish'}
            },
            
            # 不同损失函数
            'cross_entropy_loss': {
                'description': 'WE-FTT with standard cross-entropy loss',
                'params': base_params.copy(),
                'modifications': {'loss_function': 'cross_entropy'}
            },
            
            'focal_loss_static': {
                'description': 'WE-FTT with static focal loss',
                'params': base_params.copy(),
                'modifications': {'loss_function': 'focal_loss', 'adaptive_gamma': False}
            },
            
            # 不同归一化方法
            'layer_norm': {
                'description': 'WE-FTT with layer normalization',
                'params': base_params.copy(),
                'modifications': {'normalization': 'layer_norm', 'head_normalization': 'layer_norm'}
            },
            
            # 移除特征投影
            'no_feature_projection': {
                'description': 'WE-FTT without feature projection layer',
                'params': base_params.copy(),
                'modifications': {'use_feature_projection': False}
            },
            
            # 移除权重投影
            'no_weight_projection': {
                'description': 'WE-FTT without weight projection layer',
                'params': base_params.copy(),
                'modifications': {'use_weight_projection': False}
            },
            
            # 不同融合策略
            'additive_fusion': {
                'description': 'WE-FTT with additive feature-weight fusion',
                'params': base_params.copy(),
                'modifications': {'fusion_strategy': 'additive'}
            },
            
            'multiplicative_fusion': {
                'description': 'WE-FTT with multiplicative feature-weight fusion',
                'params': base_params.copy(),
                'modifications': {'fusion_strategy': 'multiplicative'}
            },
            
            # 不同模型大小
            'small_model': {
                'description': 'Smaller WE-FTT model',
                'params': base_params.copy(),
                'modifications': {
                    'hidden_dim': 256,
                    'ffn_hidden_dim': 512,
                    'n_layers': 4,
                    'n_heads': 4
                }
            },
            
            'large_model': {
                'description': 'Larger WE-FTT model',
                'params': base_params.copy(),
                'modifications': {
                    'hidden_dim': 768,
                    'ffn_hidden_dim': 2048,
                    'n_layers': 8,
                    'n_heads': 12
                }
            },
            
            # 不同dropout率
            'low_dropout': {
                'description': 'WE-FTT with low dropout rate',
                'params': base_params.copy(),
                'modifications': {'dropout_rate': 0.05}
            },
            
            'high_dropout': {
                'description': 'WE-FTT with high dropout rate',
                'params': base_params.copy(),
                'modifications': {'dropout_rate': 0.3}
            },
            
            # 不同学习率调度
            'constant_lr': {
                'description': 'WE-FTT with constant learning rate',
                'params': base_params.copy(),
                'modifications': {'scheduler_type': 'constant'}
            },
            
            'step_lr': {
                'description': 'WE-FTT with step learning rate decay',
                'params': base_params.copy(),
                'modifications': {'scheduler_type': 'step'}
            }
        }
        
        # 应用修改到参数
        for exp_name, exp_config in experiments.items():
            exp_config['params'].update(exp_config['modifications'])
        
        return experiments
    
    def run_single_experiment(
        self, 
        exp_name: str, 
        exp_config: Dict[str, Any],
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        test_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """运行单个消融实验"""
        
        logger.info(f"Running experiment: {exp_name}")
        logger.info(f"Description: {exp_config['description']}")
        
        try:
            # 创建模型
            model = create_we_ftt_model(
                num_features=len(self.base_config.COLUMNS_FEATURES),
                num_classes=2,
                config=exp_config['params'],
                use_weight_enhancement=exp_config['params'].get('use_weight_enhancement', True)
            )
            model.to(self.device)
            
            # 准备数据加载器
            from train import ModelTrainer
            
            trainer_config = {
                'model_name': 'we_ftt',
                'model_params': exp_config['params'],
                'feature_columns': self.base_config.COLUMNS_FEATURES,
                'weight_columns': self.base_config.COLUMNS_WEIGHTS,
                'label_column': self.base_config.LABEL_COLUMN,
                'num_epochs': self.config.get('num_epochs', 20),
                'batch_size': self.config.get('batch_size', 32),
                'random_seed': self.config.get('random_seed', 42),
                'optimizer_params': {
                    'type': 'adamw',
                    'lr': self.config.get('learning_rate', 0.001),
                    'weight_decay': 0.01
                },
                'scheduler_params': {
                    'type': exp_config['params'].get('scheduler_type', 'cosine'),
                    'warmup_steps': 100,
                    't_total': self.config.get('num_epochs', 20) * 100  # 假设每个epoch 100步
                }
            }
            
            # 训练模型
            trainer = ModelTrainer(trainer_config)
            trainer.model = model  # 使用我们创建的模型
            trainer.prepare_data(train_data, val_data, test_data)
            
            # 执行训练
            results = trainer.train()
            
            # 保存实验结果
            exp_output_dir = self.output_dir / exp_name
            exp_output_dir.mkdir(exist_ok=True)
            
            with open(exp_output_dir / 'results.json', 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            with open(exp_output_dir / 'config.json', 'w') as f:
                json.dump(exp_config, f, indent=2)
            
            # 保存模型
            torch.save(model.state_dict(), exp_output_dir / 'model.pth')
            
            logger.info(f"Experiment {exp_name} completed successfully")
            
            return {
                'experiment': exp_name,
                'description': exp_config['description'],
                'results': results,
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Experiment {exp_name} failed: {e}")
            return {
                'experiment': exp_name,
                'description': exp_config['description'],
                'error': str(e),
                'status': 'failed'
            }
    
    def run_all_experiments(self, parallel: bool = False) -> Dict[str, Any]:
        """运行所有消融实验"""
        
        logger.info("Loading data for ablation study...")
        
        # 加载数据
        dataset_creator = DatasetCreator()
        train_data, val_data, test_data = dataset_creator.create_training_datasets()
        
        logger.info(f"Data loaded: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
        
        # 定义实验
        experiments = self.define_ablation_experiments()
        logger.info(f"Defined {len(experiments)} ablation experiments")
        
        # 运行实验
        results = {}
        
        if parallel:
            # 并行执行实验
            logger.info("Running experiments in parallel...")
            max_workers = min(len(experiments), mp.cpu_count())
            
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_to_exp = {
                    executor.submit(
                        self.run_single_experiment, 
                        exp_name, exp_config,
                        train_data, val_data, test_data
                    ): exp_name
                    for exp_name, exp_config in experiments.items()
                }
                
                for future in as_completed(future_to_exp):
                    exp_name = future_to_exp[future]
                    try:
                        result = future.result()
                        results[exp_name] = result
                    except Exception as e:
                        logger.error(f"Experiment {exp_name} failed: {e}")
                        results[exp_name] = {
                            'experiment': exp_name,
                            'error': str(e),
                            'status': 'failed'
                        }
        else:
            # 串行执行实验
            logger.info("Running experiments sequentially...")
            for exp_name, exp_config in experiments.items():
                result = self.run_single_experiment(
                    exp_name, exp_config, train_data, val_data, test_data
                )
                results[exp_name] = result
        
        # 生成汇总报告
        summary = self._generate_summary_report(results)
        
        # 保存完整结果
        with open(self.output_dir / 'ablation_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        with open(self.output_dir / 'ablation_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info("All ablation experiments completed")
        return results
    
    def _generate_summary_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """生成汇总报告"""
        
        summary = {
            'total_experiments': len(results),
            'successful_experiments': sum(1 for r in results.values() if r['status'] == 'success'),
            'failed_experiments': sum(1 for r in results.values() if r['status'] == 'failed'),
            'experiments': {}
        }
        
        # 提取关键指标
        for exp_name, exp_result in results.items():
            if exp_result['status'] == 'success':
                test_results = exp_result['results'].get('test_results', {})
                summary['experiments'][exp_name] = {
                    'description': exp_result['description'],
                    'accuracy': test_results.get('accuracy', 0.0),
                    'f1_score': test_results.get('f1_score', 0.0),
                    'matthews_corrcoef': test_results.get('matthews_corrcoef', 0.0),
                    'cohen_kappa': test_results.get('cohen_kappa', 0.0)
                }
            else:
                summary['experiments'][exp_name] = {
                    'description': exp_result['description'],
                    'status': 'failed',
                    'error': exp_result.get('error', 'Unknown error')
                }
        
        # 排序（按准确率）
        successful_exps = {
            k: v for k, v in summary['experiments'].items() 
            if v.get('accuracy') is not None
        }
        
        if successful_exps:
            sorted_exps = sorted(
                successful_exps.items(), 
                key=lambda x: x[1]['accuracy'], 
                reverse=True
            )
            summary['ranking_by_accuracy'] = [
                {'experiment': exp, 'accuracy': info['accuracy']} 
                for exp, info in sorted_exps
            ]
            
            # 最佳和最差实验
            summary['best_experiment'] = sorted_exps[0]
            summary['worst_experiment'] = sorted_exps[-1]
        
        return summary


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Run ablation study for WE-FTT project')
    
    parser.add_argument('--output_dir', type=str, default='results/ablation_study',
                       help='Output directory for ablation results')
    
    parser.add_argument('--config_file', type=str, default=None,
                       help='Path to configuration file')
    
    parser.add_argument('--parallel', action='store_true',
                       help='Run experiments in parallel')
    
    parser.add_argument('--num_epochs', type=int, default=10,
                       help='Number of training epochs (reduced for ablation)')
    
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    
    parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    parser.add_argument('--experiments', type=str, nargs='*', default=None,
                       help='Specific experiments to run (default: all)')
    
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID')
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 设置日志
    setup_logging()
    
    # 设置GPU
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        logger.info(f"Using GPU {args.gpu}")
    else:
        logger.info("Using CPU")
    
    # 加载配置
    if args.config_file:
        with open(args.config_file, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {
            'output_dir': args.output_dir,
            'num_epochs': args.num_epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'random_seed': args.random_seed
        }
    
    # 覆盖命令行参数
    config.update({
        'output_dir': args.output_dir,
        'num_epochs': args.num_epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'random_seed': args.random_seed
    })
    
    try:
        # 初始化并运行消融实验
        ablation = AblationStudy(config)
        results = ablation.run_all_experiments(parallel=args.parallel)
        
        # 打印摘要
        logger.info("=" * 60)
        logger.info("ABLATION STUDY SUMMARY")
        logger.info("=" * 60)
        
        summary_file = Path(args.output_dir) / 'ablation_summary.json'
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                summary = json.load(f)
            
            logger.info(f"Total experiments: {summary['total_experiments']}")
            logger.info(f"Successful: {summary['successful_experiments']}")
            logger.info(f"Failed: {summary['failed_experiments']}")
            
            if 'best_experiment' in summary:
                best_exp = summary['best_experiment']
                logger.info(f"Best experiment: {best_exp[0]} (Accuracy: {best_exp[1]['accuracy']:.4f})")
            
            if 'ranking_by_accuracy' in summary:
                logger.info("\nTop 5 experiments by accuracy:")
                for i, exp in enumerate(summary['ranking_by_accuracy'][:5]):
                    logger.info(f"  {i+1}. {exp['experiment']}: {exp['accuracy']:.4f}")
        
        logger.info("=" * 60)
        logger.info("Ablation study completed successfully!")
        
    except Exception as e:
        logger.error(f"Ablation study failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()