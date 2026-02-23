#!/usr/bin/env python3
"""
Data preprocessing pipeline script for WE-FTT project.

This script runs the complete data preprocessing pipeline including:
1. Data loading and cleaning
2. Surface classification and sampling
3. K-means clustering for discretization
4. Association rule mining (Apriori, Eclat, FP-Growth)
5. Feature weight calculation

Usage:
    python scripts/run_preprocessing.py --input_data data/raw/13-23EQ.csv --output_dir data/processed
    python scripts/run_preprocessing.py --config_file configs/preprocessing.yaml
"""

import argparse
import logging
import os
import sys
import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from config import get_config, DataProcessingConfig, Config
from data_processing import DatasetCreator, process_raw_data
from association_mining import KnowledgeMiner, run_knowledge_mining
from utils import setup_logging, set_random_seeds


logger = logging.getLogger(__name__)


class PreprocessingPipeline:
    """数据预处理流水线"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.processing_config = DataProcessingConfig()
        
        # 设置随机种子
        set_random_seeds(config.get('random_seed', 42))
        
        # 输入输出路径
        self.input_data_path = config['input_data_path']
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized preprocessing pipeline")
        logger.info(f"Input data: {self.input_data_path}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def run_complete_pipeline(self) -> Dict[str, Any]:
        """运行完整的预处理流水线"""
        logger.info("Starting complete preprocessing pipeline...")
        
        results = {}
        
        # 步骤1: 基础数据处理
        logger.info("Step 1: Basic data processing...")
        train_data, val_data, test_data = self._run_basic_processing()
        results['data_processing'] = {
            'train_size': len(train_data),
            'val_size': len(val_data),
            'test_size': len(test_data)
        }
        
        # 步骤2: 知识挖掘
        logger.info("Step 2: Knowledge mining...")
        mining_results = self._run_knowledge_mining(train_data)
        results['knowledge_mining'] = mining_results
        
        # 步骤3: 应用权重到数据集
        logger.info("Step 3: Applying weights to datasets...")
        weighted_datasets = self._apply_weights_to_datasets(
            train_data, val_data, test_data, mining_results['feature_weights']
        )
        results['weighted_datasets'] = weighted_datasets
        
        # 步骤4: 保存最终数据集
        logger.info("Step 4: Saving final datasets...")
        self._save_final_datasets(weighted_datasets)
        
        # 步骤5: 生成数据统计报告
        logger.info("Step 5: Generating data statistics report...")
        stats_report = self._generate_statistics_report(weighted_datasets, mining_results)
        results['statistics'] = stats_report
        
        # 保存完整结果
        self._save_pipeline_results(results)
        
        logger.info("Preprocessing pipeline completed successfully!")
        return results
    
    def _run_basic_processing(self) -> tuple:
        """运行基础数据处理"""
        dataset_creator = DatasetCreator(self.processing_config)
        
        # 创建训练数据集
        train_df, val_df, test_df = dataset_creator.create_training_datasets(
            data_path=self.input_data_path,
            save_processed=False  # 我们稍后会保存最终版本
        )
        
        logger.info(f"Basic processing completed: "
                   f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        return train_df, val_df, test_df
    
    def _run_knowledge_mining(self, train_data: pd.DataFrame) -> Dict[str, Any]:
        """运行知识挖掘"""
        miner = KnowledgeMiner(self.processing_config)
        
        # 执行知识挖掘
        mining_results = miner.mine_knowledge(
            data=train_data,
            features=self.processing_config.COLUMNS_FEATURES,
            label_column=self.processing_config.LABEL_COLUMN,
            output_dir=str(self.output_dir / 'knowledge_mining')
        )
        
        logger.info(f"Knowledge mining completed: "
                   f"Found {len(mining_results['frequent_itemsets'])} frequent itemset levels, "
                   f"{len(mining_results['association_rules'])} rules")
        
        return mining_results
    
    def _apply_weights_to_datasets(
        self, 
        train_data: pd.DataFrame, 
        val_data: pd.DataFrame, 
        test_data: pd.DataFrame,
        feature_weights: Dict[str, float]
    ) -> Dict[str, pd.DataFrame]:
        """将计算出的权重应用到数据集"""
        
        # 为每个特征添加权重列
        for dataset_name, dataset in [('train', train_data), ('val', val_data), ('test', test_data)]:
            for feature, weight in feature_weights.items():
                if feature in dataset.columns:
                    weight_col_name = f"{feature}_cluster_labels_weight"
                    dataset[weight_col_name] = weight
                    logger.debug(f"Added weight {weight:.4f} for {feature} in {dataset_name} dataset")
        
        weighted_datasets = {
            'train': train_data,
            'val': val_data,
            'test': test_data
        }
        
        logger.info("Applied feature weights to all datasets")
        return weighted_datasets
    
    def _save_final_datasets(self, datasets: Dict[str, pd.DataFrame]):
        """保存最终数据集"""
        for name, dataset in datasets.items():
            # 保存为parquet格式（更高效）
            parquet_path = self.output_dir / f"{name}_dataset.parquet"
            dataset.to_parquet(parquet_path, index=False)
            
            # 也保存为CSV格式（便于查看）
            csv_path = self.output_dir / f"{name}_dataset.csv"
            dataset.to_csv(csv_path, index=False)
            
            logger.info(f"Saved {name} dataset: {len(dataset)} samples, {len(dataset.columns)} columns")
    
    def _generate_statistics_report(
        self, 
        datasets: Dict[str, pd.DataFrame],
        mining_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """生成数据统计报告"""
        
        stats = {}
        
        # 数据集统计
        for name, dataset in datasets.items():
            stats[f"{name}_dataset"] = {
                'num_samples': len(dataset),
                'num_features': len(self.processing_config.COLUMNS_FEATURES),
                'num_weight_columns': len([col for col in dataset.columns if 'weight' in col]),
                'label_distribution': dataset[self.processing_config.LABEL_COLUMN].value_counts().to_dict(),
                'feature_stats': dataset[self.processing_config.COLUMNS_FEATURES].describe().to_dict()
            }
        
        # 聚类统计
        stats['clustering'] = {
            'num_clusters_per_feature': self.processing_config.KMEANS_CONFIG['n_clusters'],
            'features_clustered': list(mining_results['cluster_info'].keys())
        }
        
        # 关联规则统计
        stats['association_rules'] = {
            'num_frequent_itemsets': sum(
                len(itemsets) for itemsets in mining_results['frequent_itemsets'].values()
            ),
            'num_association_rules': len(mining_results['association_rules']),
            'min_support': self.processing_config.APRIORI_CONFIG['min_support'],
            'min_confidence': self.processing_config.APRIORI_CONFIG['min_confidence']
        }
        
        # 特征权重统计
        weights = mining_results['feature_weights']
        stats['feature_weights'] = {
            'weight_range': [min(weights.values()), max(weights.values())],
            'mean_weight': sum(weights.values()) / len(weights),
            'weights_by_feature': weights
        }
        
        # 保存统计报告
        stats_file = self.output_dir / 'statistics_report.json'
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        
        logger.info(f"Statistics report saved to {stats_file}")
        return stats
    
    def _save_pipeline_results(self, results: Dict[str, Any]):
        """保存流水线结果"""
        results_file = self.output_dir / 'preprocessing_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # 保存配置
        config_file = self.output_dir / 'preprocessing_config.json'
        with open(config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        logger.info(f"Pipeline results saved to {results_file}")


def validate_input_data(data_path: str) -> bool:
    """验证输入数据"""
    if not os.path.exists(data_path):
        logger.error(f"Input data file not found: {data_path}")
        return False
    
    try:
        # 尝试读取数据文件
        if data_path.endswith('.csv'):
            df = pd.read_csv(data_path, nrows=5)  # 只读前5行进行验证
        elif data_path.endswith('.parquet'):
            df = pd.read_parquet(data_path, nrows=5)
        else:
            logger.error(f"Unsupported file format: {data_path}")
            return False
        
        logger.info(f"Input data validation passed: {len(df.columns)} columns")
        return True
        
    except Exception as e:
        logger.error(f"Error reading input data: {e}")
        return False


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Run data preprocessing pipeline for WE-FTT project')
    
    parser.add_argument('--input_data', type=str, required=True,
                       help='Path to input data file')
    
    parser.add_argument('--output_dir', type=str, default='data/processed',
                       help='Output directory for processed data')
    
    parser.add_argument('--config_file', type=str, default=None,
                       help='Path to configuration file')
    
    parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    parser.add_argument('--min_support', type=float, default=0.1,
                       help='Minimum support for association rule mining')
    
    parser.add_argument('--min_confidence', type=float, default=0.7,
                       help='Minimum confidence for association rule mining')
    
    parser.add_argument('--n_clusters', type=int, default=5,
                       help='Number of clusters for K-means')
    
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Test set size ratio')
    
    parser.add_argument('--val_size', type=float, default=0.2,
                       help='Validation set size ratio')
    
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 设置日志
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=log_level)
    
    logger.info("Starting data preprocessing pipeline...")
    
    # 验证输入数据
    if not validate_input_data(args.input_data):
        sys.exit(1)
    
    # 构建配置
    if args.config_file:
        import yaml
        with open(args.config_file, 'r') as f:
            config = yaml.safe_load(f)
    else:
        # 使用命令行参数构建配置
        config = {
            'input_data_path': args.input_data,
            'output_dir': args.output_dir,
            'random_seed': args.random_seed,
            'kmeans_config': {
                'n_clusters': args.n_clusters
            },
            'apriori_config': {
                'min_support': args.min_support,
                'min_confidence': args.min_confidence
            },
            'sampling_config': {
                'test_size': args.test_size,
                'val_size': args.val_size
            }
        }
    
    # 确保必要的参数存在
    config.setdefault('input_data_path', args.input_data)
    config.setdefault('output_dir', args.output_dir)
    config.setdefault('random_seed', args.random_seed)
    
    try:
        # 初始化并运行预处理流水线
        pipeline = PreprocessingPipeline(config)
        results = pipeline.run_complete_pipeline()
        
        # 打印摘要
        logger.info("=" * 50)
        logger.info("PREPROCESSING PIPELINE SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Input data: {args.input_data}")
        logger.info(f"Output directory: {args.output_dir}")
        logger.info(f"Training samples: {results['data_processing']['train_size']}")
        logger.info(f"Validation samples: {results['data_processing']['val_size']}")
        logger.info(f"Test samples: {results['data_processing']['test_size']}")
        logger.info(f"Association rules found: {len(results['knowledge_mining']['association_rules'])}")
        logger.info(f"Feature weights calculated: {len(results['knowledge_mining']['feature_weights'])}")
        logger.info("=" * 50)
        
        logger.info("Preprocessing pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Preprocessing pipeline failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()