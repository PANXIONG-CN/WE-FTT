"""
Data processing module for WE-FTT project.

This module contains functions for data loading, preprocessing, sampling,
and preparation for model training.
"""

import os
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, List, Optional, Dict, Any
import logging

from .config import Config, DataProcessingConfig


logger = logging.getLogger(__name__)


class DataLoader:
    """数据加载器类，处理各种格式的数据文件"""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
    
    def load_earthquake_catalog(self) -> pd.DataFrame:
        """加载地震目录数据"""
        try:
            df = pd.read_csv(self.config.EARTHQUAKE_CATALOG)
            logger.info(f"Loaded earthquake catalog with {len(df)} records")
            return df
        except Exception as e:
            logger.error(f"Error loading earthquake catalog: {e}")
            raise
    
    def load_training_data(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """加载训练数据"""
        if file_path is None:
            file_path = self.config.TRAINING_DATASET
        
        try:
            if str(file_path).endswith('.parquet'):
                df = pd.read_parquet(file_path)
            elif str(file_path).endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
            
            logger.info(f"Loaded training data with shape {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error loading training data: {e}")
            raise
    
    def load_processed_data(self, data_type: str) -> List[Dict]:
        """加载已处理的数据（如聚类结果、频繁项集等）"""
        processed_dir = self.config.DATA_PROCESSED_DIR
        
        if data_type == "cluster_info":
            cluster_files = list(processed_dir.glob("*_cluster_info.json"))
            return [pd.read_json(f) for f in cluster_files]
        elif data_type == "frequent_itemsets":
            itemset_files = list(processed_dir.glob("*_freqItemsets_*.json"))
            return [pd.read_json(f) for f in itemset_files]
        else:
            raise ValueError(f"Unknown data type: {data_type}")


class DataPreprocessor:
    """数据预处理器类"""
    
    def __init__(self, config: DataProcessingConfig = None):
        self.config = config or DataProcessingConfig()
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """验证数据完整性"""
        required_columns = (
            self.config.COLUMNS_FEATURES + 
            self.config.COLUMNS_WEIGHTS + 
            [self.config.LABEL_COLUMN]
        )
        
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False
        
        # 检查空值
        null_counts = df[required_columns].isnull().sum()
        if null_counts.any():
            logger.warning(f"Found null values: {null_counts[null_counts > 0]}")
        
        logger.info("Data validation passed")
        return True
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """数据清洗"""
        # 移除重复行
        initial_len = len(df)
        df = df.drop_duplicates()
        if len(df) < initial_len:
            logger.info(f"Removed {initial_len - len(df)} duplicate rows")
        
        # 处理异常值（使用IQR方法）
        for col in self.config.COLUMNS_FEATURES:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
            if outliers.any():
                logger.info(f"Found {outliers.sum()} outliers in {col}")
                # 可以选择移除或修正异常值
                # df = df[~outliers]  # 移除异常值
                # 或者用边界值替换
                df.loc[df[col] < lower_bound, col] = lower_bound
                df.loc[df[col] > upper_bound, col] = upper_bound
        
        return df
    
    def normalize_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """特征标准化"""
        df_normalized = df.copy()
        
        if fit:
            df_normalized[self.config.COLUMNS_FEATURES] = self.scaler.fit_transform(
                df[self.config.COLUMNS_FEATURES]
            )
        else:
            df_normalized[self.config.COLUMNS_FEATURES] = self.scaler.transform(
                df[self.config.COLUMNS_FEATURES]
            )
        
        logger.info("Features normalized")
        return df_normalized
    
    def encode_labels(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """标签编码"""
        df_encoded = df.copy()
        
        if fit:
            df_encoded[self.config.LABEL_COLUMN] = self.label_encoder.fit_transform(
                df[self.config.LABEL_COLUMN]
            )
        else:
            df_encoded[self.config.LABEL_COLUMN] = self.label_encoder.transform(
                df[self.config.LABEL_COLUMN]
            )
        
        logger.info(f"Labels encoded: {self.label_encoder.classes_}")
        return df_encoded


class DataSplitter:
    """数据分割器类"""
    
    def __init__(self, config: DataProcessingConfig = None):
        self.config = config or DataProcessingConfig()
    
    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """将数据分割为训练集、验证集和测试集"""
        X = df[self.config.COLUMNS_FEATURES + self.config.COLUMNS_WEIGHTS]
        y = df[self.config.LABEL_COLUMN]
        
        # 首先分离出测试集
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=self.config.SAMPLING_CONFIG["test_size"],
            random_state=self.config.SAMPLING_CONFIG["random_state"],
            stratify=y if self.config.SAMPLING_CONFIG["stratify"] else None
        )
        
        # 然后从剩余数据中分离出验证集
        val_size_adjusted = self.config.SAMPLING_CONFIG["val_size"] / (
            1 - self.config.SAMPLING_CONFIG["test_size"]
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=self.config.SAMPLING_CONFIG["random_state"],
            stratify=y_temp if self.config.SAMPLING_CONFIG["stratify"] else None
        )
        
        train_df = pd.concat([X_train, y_train], axis=1)
        val_df = pd.concat([X_val, y_val], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)
        
        logger.info(f"Data split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        return train_df, val_df, test_df


class DatasetCreator:
    """数据集创建器，整合所有预处理步骤"""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.loader = DataLoader(config)
        self.preprocessor = DataPreprocessor(config)
        self.splitter = DataSplitter(config)
    
    def create_training_datasets(
        self, 
        data_path: Optional[str] = None,
        save_processed: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """创建完整的训练数据集"""
        
        # 1. 加载数据
        logger.info("Starting data processing pipeline...")
        df = self.loader.load_training_data(data_path)
        
        # 2. 验证数据
        if not self.preprocessor.validate_data(df):
            raise ValueError("Data validation failed")
        
        # 3. 数据清洗
        df = self.preprocessor.clean_data(df)
        
        # 4. 特征标准化
        df = self.preprocessor.normalize_features(df, fit=True)
        
        # 5. 标签编码
        df = self.preprocessor.encode_labels(df, fit=True)
        
        # 6. 数据分割
        train_df, val_df, test_df = self.splitter.split_data(df)
        
        # 7. 保存处理后的数据
        if save_processed:
            output_dir = self.config.DATA_PROCESSED_DIR
            train_df.to_parquet(output_dir / "train_dataset.parquet")
            val_df.to_parquet(output_dir / "val_dataset.parquet")
            test_df.to_parquet(output_dir / "test_dataset.parquet")
            logger.info("Processed datasets saved")
        
        logger.info("Data processing pipeline completed")
        return train_df, val_df, test_df


def process_raw_data(input_file: str, output_dir: str, config: Config = None) -> None:
    """
    处理原始数据的便捷函数
    
    Args:
        input_file: 输入数据文件路径
        output_dir: 输出目录路径
        config: 配置对象
    """
    config = config or Config()
    creator = DatasetCreator(config)
    
    try:
        train_df, val_df, test_df = creator.create_training_datasets(input_file)
        
        # 保存数据集统计信息
        stats = {
            "train_size": len(train_df),
            "val_size": len(val_df),
            "test_size": len(test_df),
            "feature_columns": config.COLUMNS_FEATURES,
            "weight_columns": config.COLUMNS_WEIGHTS,
            "label_distribution": train_df[config.LABEL_COLUMN].value_counts().to_dict()
        }
        
        import json
        with open(f"{output_dir}/dataset_stats.json", "w") as f:
            json.dump(stats, f, indent=2)
        
        logger.info("Raw data processing completed successfully")
        
    except Exception as e:
        logger.error(f"Error processing raw data: {e}")
        raise