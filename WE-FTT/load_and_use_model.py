#!/usr/bin/env python3
"""
加载和使用训练好的WE-FTT模型

示例展示：
1. 如何加载保存的模型
2. 如何进行预测
3. 如何评估模型性能
"""

import sys
import torch
import pandas as pd
import numpy as np
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.models.we_ftt import create_we_ftt_model
from src.config import WEFTTConfig
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def load_trained_model(checkpoint_path):
    """
    加载训练好的模型
    
    Args:
        checkpoint_path: 模型checkpoint文件路径 (best_model.pth)
    
    Returns:
        model: 加载好的模型
        config: 模型配置
    """
    print(f"Loading model from: {checkpoint_path}")
    
    # 加载checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 获取配置
    config = checkpoint['config']
    num_features = len(config['feature_columns'])
    num_classes = config.get('num_classes', 2)
    
    # 创建模型
    model = create_we_ftt_model(
        num_features=num_features,
        num_classes=num_classes,
        config=config.get('model_params', {}),
        use_weight_enhancement=True
    )
    
    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # 设置为评估模式
    
    print(f"✓ Model loaded successfully!")
    print(f"  - Epoch: {checkpoint['epoch']}")
    print(f"  - Val Loss: {checkpoint['val_loss']:.4f}")
    print(f"  - Val Accuracy: {checkpoint['val_accuracy']:.4f}")
    
    return model, config


def predict(model, features, weights=None, device='cpu'):
    """
    使用模型进行预测
    
    Args:
        model: WE-FTT模型
        features: 特征数据 (numpy array或tensor)
        weights: 权重数据 (可选)
        device: 计算设备
    
    Returns:
        predictions: 预测类别
        probabilities: 预测概率
    """
    model = model.to(device)
    model.eval()
    
    # 转换为tensor
    if isinstance(features, np.ndarray):
        features = torch.from_numpy(features).float()
    if weights is not None and isinstance(weights, np.ndarray):
        weights = torch.from_numpy(weights).float()
    
    features = features.to(device)
    if weights is not None:
        weights = weights.to(device)
    
    with torch.no_grad():
        if weights is not None:
            outputs = model(features, weights)
        else:
            outputs = model(features)
        
        probabilities = torch.softmax(outputs, dim=1)
        predictions = torch.argmax(probabilities, dim=1)
    
    return predictions.cpu().numpy(), probabilities.cpu().numpy()


def evaluate_model(model, test_data, config, device='cpu'):
    """
    评估模型性能
    
    Args:
        model: WE-FTT模型
        test_data: 测试数据 (DataFrame)
        config: 模型配置
        device: 计算设备
    
    Returns:
        metrics: 评估指标字典
    """
    feature_cols = config['feature_columns']
    weight_cols = config.get('weight_columns', [])
    label_col = config['label_column']
    
    X_test = test_data[feature_cols].values
    y_test = test_data[label_col].values
    
    if weight_cols and len(weight_cols) > 0:
        W_test = test_data[weight_cols].values
    else:
        W_test = None
    
    # 批量预测
    batch_size = 10000
    all_preds = []
    all_probs = []
    
    for i in range(0, len(X_test), batch_size):
        batch_features = X_test[i:i+batch_size]
        batch_weights = W_test[i:i+batch_size] if W_test is not None else None
        
        preds, probs = predict(model, batch_features, batch_weights, device)
        all_preds.extend(preds)
        all_probs.extend(probs)
    
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    # 计算指标
    accuracy = accuracy_score(y_test, all_preds)
    
    print("\n" + "="*60)
    print("Model Evaluation Results")
    print("="*60)
    print(f"\nAccuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, all_preds, 
                                target_names=['Non-earthquake', 'Earthquake']))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, all_preds))
    
    return {
        'accuracy': accuracy,
        'predictions': all_preds,
        'probabilities': all_probs
    }


def main():
    """主函数示例"""
    
    # 1. 设置路径
    checkpoint_path = 'results/we_ftt/best_model.pth'
    
    if not Path(checkpoint_path).exists():
        print(f"Error: Checkpoint file not found: {checkpoint_path}")
        print("Please train the model first using:")
        print("  bash quick_start_training.sh")
        return
    
    # 2. 加载模型
    model, config = load_trained_model(checkpoint_path)
    
    # 3. 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # 4. 示例：加载测试数据并预测
    print("\n" + "="*60)
    print("Example: Load test data and make predictions")
    print("="*60)
    
    # 这里你需要替换成实际的测试数据路径
    test_data_path = 'data/processed/training_data_labeled.parquet'
    
    if Path(test_data_path).exists():
        print(f"\nLoading test data from: {test_data_path}")
        # 为了演示，只加载一小部分数据
        test_data = pd.read_parquet(test_data_path)
        test_sample = test_data.sample(n=min(10000, len(test_data)), random_state=42)
        
        print(f"Test sample size: {len(test_sample)}")
        
        # 评估模型
        metrics = evaluate_model(model, test_sample, config, device)
        
        print(f"\n✓ Evaluation completed!")
        print(f"  Accuracy on test sample: {metrics['accuracy']:.4f}")
    else:
        print(f"\nTest data not found: {test_data_path}")
        print("Skipping evaluation example.")
    
    # 5. 示例：单个样本预测
    print("\n" + "="*60)
    print("Example: Single sample prediction")
    print("="*60)
    
    # 创建示例特征（你需要替换成实际数据）
    sample_features = np.random.randn(1, len(config['feature_columns']))
    sample_weights = np.ones((1, len(config.get('weight_columns', []))))
    
    preds, probs = predict(model, sample_features, sample_weights, device)
    
    print(f"\nSample prediction:")
    print(f"  Predicted class: {preds[0]} ({'Earthquake' if preds[0]==1 else 'Non-earthquake'})")
    print(f"  Probabilities: Non-earthquake={probs[0][0]:.4f}, Earthquake={probs[0][1]:.4f}")
    
    print("\n" + "="*60)
    print("Done!")
    print("="*60)


if __name__ == '__main__':
    main()
