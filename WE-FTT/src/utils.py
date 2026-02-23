"""
Utility functions for WE-FTT project.

This module contains helper functions for logging, reproducibility,
model saving/loading, and other common utilities.
"""

import os
import logging
import random
import numpy as np
import torch
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Union
from datetime import datetime


def setup_logging(
    level: Union[int, str] = logging.INFO,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> None:
    """
    设置日志配置
    
    Args:
        level: 日志级别
        log_file: 日志文件路径（可选）
        format_string: 日志格式字符串（可选）
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # 配置根logger
    logging.basicConfig(
        level=level,
        format=format_string,
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # 如果指定了日志文件，添加文件处理器
    if log_file:
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(format_string)
        file_handler.setFormatter(file_formatter)
        
        # 获取根logger并添加文件处理器
        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)


def set_random_seeds(seed: int = 42) -> None:
    """
    设置所有随机种子以确保可复现性
    
    Args:
        seed: 随机种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # 确保CUDA操作的确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 设置环境变量
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    logging.info(f"Random seeds set to {seed}")


def save_model(
    model: torch.nn.Module,
    save_path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
    loss: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    保存模型和相关信息
    
    Args:
        model: PyTorch模型
        save_path: 保存路径
        optimizer: 优化器（可选）
        epoch: 当前epoch（可选）
        loss: 当前loss（可选）
        metadata: 额外的元数据（可选）
    """
    save_dir = Path(save_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'timestamp': datetime.now().isoformat(),
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    if epoch is not None:
        checkpoint['epoch'] = epoch
    
    if loss is not None:
        checkpoint['loss'] = loss
    
    if metadata is not None:
        checkpoint['metadata'] = metadata
    
    torch.save(checkpoint, save_path)
    logging.info(f"Model saved to {save_path}")


def load_model(
    model: torch.nn.Module,
    load_path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    加载模型和相关信息
    
    Args:
        model: PyTorch模型
        load_path: 加载路径
        optimizer: 优化器（可选）
        device: 设备（可选）
    
    Returns:
        包含加载信息的字典
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint = torch.load(load_path, map_location=device)
    
    # 加载模型状态
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 加载优化器状态（如果提供）
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    logging.info(f"Model loaded from {load_path}")
    
    return {
        'epoch': checkpoint.get('epoch'),
        'loss': checkpoint.get('loss'),
        'timestamp': checkpoint.get('timestamp'),
        'metadata': checkpoint.get('metadata')
    }


def save_json(data: Dict[str, Any], file_path: str) -> None:
    """
    保存数据为JSON格式
    
    Args:
        data: 要保存的数据
        file_path: 文件路径
    """
    save_dir = Path(file_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    
    logging.info(f"JSON data saved to {file_path}")


def load_json(file_path: str) -> Dict[str, Any]:
    """
    从JSON文件加载数据
    
    Args:
        file_path: 文件路径
    
    Returns:
        加载的数据
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    logging.info(f"JSON data loaded from {file_path}")
    return data


def save_pickle(data: Any, file_path: str) -> None:
    """
    保存数据为pickle格式
    
    Args:
        data: 要保存的数据
        file_path: 文件路径
    """
    save_dir = Path(file_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
    
    logging.info(f"Pickle data saved to {file_path}")


def load_pickle(file_path: str) -> Any:
    """
    从pickle文件加载数据
    
    Args:
        file_path: 文件路径
    
    Returns:
        加载的数据
    """
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    logging.info(f"Pickle data loaded from {file_path}")
    return data


def get_device(gpu_id: Optional[int] = None) -> torch.device:
    """
    获取计算设备
    
    Args:
        gpu_id: GPU ID（可选）
    
    Returns:
        PyTorch设备对象
    """
    if torch.cuda.is_available():
        if gpu_id is not None:
            device = torch.device(f'cuda:{gpu_id}')
        else:
            device = torch.device('cuda')
        
        logging.info(f"Using device: {device}")
        logging.info(f"GPU: {torch.cuda.get_device_name(device)}")
        logging.info(f"GPU Memory: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f} GB")
    else:
        device = torch.device('cpu')
        logging.info("CUDA not available. Using CPU.")
    
    return device


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """
    计算模型参数数量
    
    Args:
        model: PyTorch模型
    
    Returns:
        参数统计信息
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    stats = {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params
    }
    
    logging.info(f"Model parameters: {stats}")
    return stats


def create_timestamp() -> str:
    """
    创建时间戳字符串
    
    Returns:
        时间戳字符串
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    确保目录存在
    
    Args:
        path: 目录路径
    
    Returns:
        Path对象
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_file_size(file_path: Union[str, Path]) -> str:
    """
    获取文件大小的人类可读格式
    
    Args:
        file_path: 文件路径
    
    Returns:
        文件大小字符串
    """
    file_path = Path(file_path)
    if not file_path.exists():
        return "0 B"
    
    size = file_path.stat().st_size
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} PB"


def format_time(seconds: float) -> str:
    """
    格式化时间（秒）为人类可读格式
    
    Args:
        seconds: 秒数
    
    Returns:
        格式化的时间字符串
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


class Timer:
    """简单的计时器类"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
    
    def start(self):
        """开始计时"""
        self.start_time = datetime.now()
        logging.info("Timer started")
    
    def stop(self):
        """停止计时"""
        if self.start_time is None:
            logging.warning("Timer was not started")
            return None
        
        self.end_time = datetime.now()
        elapsed = (self.end_time - self.start_time).total_seconds()
        logging.info(f"Timer stopped. Elapsed time: {format_time(elapsed)}")
        return elapsed
    
    def elapsed(self) -> Optional[float]:
        """获取已经过的时间"""
        if self.start_time is None:
            return None
        
        current_time = datetime.now()
        elapsed = (current_time - self.start_time).total_seconds()
        return elapsed


class TensorDataset(torch.utils.data.Dataset):
    """自定义张量数据集"""
    
    def __init__(
        self, 
        dataframe,
        feature_columns,
        weight_columns=None,
        label_column='label'
    ):
        self.features = torch.FloatTensor(dataframe[feature_columns].values)
        self.labels = torch.LongTensor(dataframe[label_column].values)
        
        if weight_columns and len(weight_columns) > 0:
            self.weights = torch.FloatTensor(dataframe[weight_columns].values)
        else:
            self.weights = None
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        
        if self.weights is not None:
            weight = self.weights[idx]
            return feature, weight, label
        else:
            return feature, None, label


def log_system_info():
    """记录系统信息"""
    import platform
    import psutil
    
    logging.info("System Information:")
    logging.info(f"Platform: {platform.platform()}")
    logging.info(f"Python: {platform.python_version()}")
    logging.info(f"CPU: {platform.processor()}")
    logging.info(f"CPU Cores: {psutil.cpu_count()}")
    logging.info(f"Memory: {psutil.virtual_memory().total / 1024**3:.1f} GB")
    
    if torch.cuda.is_available():
        logging.info(f"CUDA: {torch.version.cuda}")
        logging.info(f"PyTorch: {torch.__version__}")
        logging.info(f"GPU Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            logging.info(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")


def validate_config(config: Dict[str, Any], required_keys: list) -> bool:
    """
    验证配置字典是否包含必需的键
    
    Args:
        config: 配置字典
        required_keys: 必需的键列表
    
    Returns:
        是否验证通过
    """
    missing_keys = []
    for key in required_keys:
        if key not in config:
            missing_keys.append(key)
    
    if missing_keys:
        logging.error(f"Missing required configuration keys: {missing_keys}")
        return False
    
    logging.info("Configuration validation passed")
    return True