"""
Central configuration management for WE-FTT project.
All file paths, hyperparameters, and model configurations are defined here.
"""

import os
from pathlib import Path


class Config:
    """Base configuration class containing all project settings."""
    
    # Project root directory
    PROJECT_ROOT = Path(__file__).parent.parent.absolute()
    
    # Data path configuration
    DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
    DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
    SAMPLE_DATA_DIR = DATA_PROCESSED_DIR / "sample_data"
    RESULTS_DIR = PROJECT_ROOT / "results"
    
    # Data file paths
    EARTHQUAKE_CATALOG = DATA_RAW_DIR / "earthquake_catalog.csv"
    TRAINING_DATASET = DATA_PROCESSED_DIR / "downsampled_f0t0.parquet"  # Can be modified as needed
    
    # Feature column definitions
    COLUMNS_FEATURES = [
        "BT_06_H", "BT_06_V", "BT_10_H", "BT_10_V",
        "BT_23_H", "BT_23_V", "BT_36_H", "BT_36_V",
        "BT_89_H", "BT_89_V"
    ]
    
    # Weight column definitions
    COLUMNS_WEIGHTS = [
        "BT_06_H_cluster_labels_weight", "BT_06_V_cluster_labels_weight",
        "BT_10_H_cluster_labels_weight", "BT_10_V_cluster_labels_weight",
        "BT_23_H_cluster_labels_weight", "BT_23_V_cluster_labels_weight",
        "BT_36_H_cluster_labels_weight", "BT_36_V_cluster_labels_weight",
        "BT_89_H_cluster_labels_weight", "BT_89_V_cluster_labels_weight"
    ]
    
    # Label column
    LABEL_COLUMN = "label"
    
    # Training configuration
    BATCH_SIZE = 100000
    NUM_WORKERS = 1
    MEMORY_LIMIT_GB = 100
    NUM_EPOCHS = 20
    PATIENCE = 5
    
    # Random seed
    RANDOM_SEED = 42
    
    # Hyperparameter search configuration
    NUM_TRIALS = 10


class WEFTTConfig(Config):
    """Weight-Enhanced FT-Transformer specific configuration."""
    
    # WE-FTT best hyperparameters (based on previous experimental results)
    BEST_PARAMS = {
        "learning_rate": 0.001,
        "weight_decay": 0.01,
        "dropout_rate": 0.1,
        "n_heads": 8,
        "n_layers": 6,
        "hidden_dim": 512,
        "ffn_hidden_dim": 1024,
        "use_cls_token": True,
        "normalization": "batch_norm",
        "activation": "relu",
        "prenormalization": True,
        "kv_compression_ratio": None,
        "kv_compression_sharing": None,
        "head_activation": "relu",
        "head_normalization": "batch_norm",
        "residual_dropout": 0.0,
        "use_weight_enhancement": True
    }


class BaselineConfig(Config):
    """Baseline models configuration."""
    
    # RandomForest configuration
    RANDOM_FOREST_PARAMS = {
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "random_state": RANDOM_SEED
    }
    
    # CatBoost configuration
    CATBOOST_PARAMS = {
        "iterations": 1000,
        "learning_rate": 0.1,
        "depth": 6,
        "random_seed": RANDOM_SEED,
        "verbose": False
    }
    
    # TabNet configuration
    TABNET_PARAMS = {
        "n_d": 64,
        "n_a": 64,
        "n_steps": 5,
        "gamma": 1.5,
        "n_independent": 2,
        "n_shared": 2,
        "epsilon": 1e-15,
        "momentum": 0.3,
        "clip_value": 2.0,
        "optimizer_fn": "adam",
        "optimizer_params": {"lr": 2e-2},
        "scheduler_params": {"step_size": 50, "gamma": 0.9},
        "scheduler_fn": "step_lr",
        "seed": RANDOM_SEED
    }
    
    # XGBoost configuration
    XGBOOST_PARAMS = {
        "n_estimators": 1000,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": RANDOM_SEED,
        "n_jobs": -1
    }
    
    # LightGBM configuration
    LIGHTGBM_PARAMS = {
        "n_estimators": 1000,
        "max_depth": -1,
        "learning_rate": 0.1,
        "num_leaves": 31,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
        "verbosity": -1
    }


class DataProcessingConfig(Config):
    """Data preprocessing and association mining configuration."""
    
    # K-means clustering configuration
    KMEANS_CONFIG = {
        "n_clusters": 5,  # Adjust based on actual needs
        "random_state": RANDOM_SEED,
        "n_init": 10,
        "max_iter": 300
    }
    
    # Apriori algorithm configuration
    APRIORI_CONFIG = {
        "min_support": 0.1,
        "min_confidence": 0.7,
        "min_lift": 1.0
    }
    
    # Data sampling configuration
    SAMPLING_CONFIG = {
        "test_size": 0.2,
        "val_size": 0.2,
        "random_state": RANDOM_SEED,
        "stratify": True
    }


class ExperimentConfig(Config):
    """Experiment tracking and logging configuration."""
    
    # Experiment result save paths
    MAIN_MODEL_RESULTS = RESULTS_DIR / "main_model"
    BASELINE_RESULTS = RESULTS_DIR / "baseline_models"
    ABLATION_RESULTS = RESULTS_DIR / "ablation_study"
    
    # Logging configuration
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Model saving configuration
    SAVE_BEST_MODEL = True
    SAVE_CHECKPOINTS = True
    CHECKPOINT_INTERVAL = 5  # Save every 5 epochs
    
    # Evaluation metrics
    METRICS = [
        "accuracy", "precision", "recall", "f1_score",
        "roc_auc", "matthews_corrcoef", "cohen_kappa"
    ]


def get_config(config_type: str = "base"):
    """
    Return the corresponding configuration object based on configuration type.
    
    Args:
        config_type: Configuration type ('base', 'we_ftt', 'baseline', 'data_processing', 'experiment')
    
    Returns:
        Corresponding configuration object
    """
    config_map = {
        "base": Config,
        "we_ftt": WEFTTConfig,
        "baseline": BaselineConfig,
        "data_processing": DataProcessingConfig,
        "experiment": ExperimentConfig
    }
    
    if config_type not in config_map:
        raise ValueError(f"Unknown config type: {config_type}")
    
    return config_map[config_type]()