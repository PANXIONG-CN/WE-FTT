"""
Weight-Enhanced Feature-based Transformer (WE-FTT) model implementation.

This module contains the main WE-FTT model that integrates knowledge-guided
feature weighting with a Feature-based Transformer architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Any, Tuple
import logging

from .components import (
    TransformerBlock,
    FeatureEmbedding,
    PositionalEncoding,
    DynamicFocalLoss,
    get_sinusoidal_encoding_table
)
from ..config import WEFTTConfig


logger = logging.getLogger(__name__)


class WEFTTransformerModel(nn.Module):
    """Weight-Enhanced Feature-based Transformer模型"""
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        num_features: int = 10,
        num_classes: int = 2,
        use_weight_enhancement: bool = True
    ):
        super().__init__()
        
        # 加载配置
        if config is None:
            config = WEFTTConfig.BEST_PARAMS
        
        self.config = config
        self.num_features = num_features
        self.num_classes = num_classes
        self.use_weight_enhancement = use_weight_enhancement
        
        # 模型参数
        self.d_model = config.get('hidden_dim', 512)
        self.n_heads = config.get('n_heads', 8)
        self.n_layers = config.get('n_layers', 6)
        self.d_ff = config.get('ffn_hidden_dim', 1024)
        self.dropout = config.get('dropout_rate', 0.1)
        self.activation = config.get('activation', 'relu')
        self.prenormalization = config.get('prenormalization', True)
        self.use_cls_token = config.get('use_cls_token', True)
        
        # KV压缩参数
        self.kv_compression_ratio = config.get('kv_compression_ratio', None)
        self.kv_compression_sharing = config.get('kv_compression_sharing', None)
        
        # 特征嵌入层
        self.feature_embedding = FeatureEmbedding(
            input_dim=1,  # 每个特征单独嵌入
            d_model=self.d_model,
            dropout=self.dropout
        )
        
        # 权重增强层（如果启用）
        if self.use_weight_enhancement:
            self.weight_projection = nn.Linear(1, self.d_model)
            self.weight_fusion = WeightFusionLayer(self.d_model, self.dropout)
        
        # 位置编码
        learnable_pos_enc = config.get('learnable_pos_enc', False)
        max_seq_len = num_features + (1 if self.use_cls_token else 0)
        self.positional_encoding = PositionalEncoding(
            self.d_model, max_seq_len, learnable=learnable_pos_enc
        )
        
        # CLS token（如果启用）
        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, self.d_model))
        
        # Transformer编码器层
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(
                d_model=self.d_model,
                n_heads=self.n_heads,
                d_ff=self.d_ff,
                dropout=self.dropout,
                activation=self.activation,
                prenormalization=self.prenormalization,
                kv_compression_ratio=self.kv_compression_ratio,
                kv_compression_sharing=self.kv_compression_sharing
            )
            for _ in range(self.n_layers)
        ])
        
        # 分类头
        self.classification_head = ClassificationHead(
            d_model=self.d_model,
            num_classes=num_classes,
            dropout=self.dropout,
            activation=config.get('head_activation', 'relu'),
            normalization=config.get('head_normalization', 'batch_norm')
        )
        
        # 损失函数
        self.criterion = DynamicFocalLoss(
            alpha=1.0,
            gamma=2.0,
            adaptive_gamma=True
        )
        
        # 权重初始化
        self._init_weights()
        
        logger.info(f"WE-FTT model initialized with {self._count_parameters()} parameters")
    
    def _init_weights(self):
        """权重初始化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
    
    def _count_parameters(self) -> int:
        """计算模型参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(
        self, 
        features: torch.Tensor, 
        weights: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            features: 输入特征 [batch_size, num_features]
            weights: 特征权重 [batch_size, num_features] (可选)
            mask: 注意力掩码 (可选)
        
        Returns:
            分类输出 [batch_size, num_classes]
        """
        batch_size = features.size(0)
        
        # 1. 特征嵌入
        # 将特征重塑为 [batch_size, num_features, 1] 用于独立嵌入
        features_reshaped = features.unsqueeze(-1)  # [batch_size, num_features, 1]
        
        # 每个特征独立嵌入
        embedded_features = []
        for i in range(self.num_features):
            feature_i = features_reshaped[:, i:i+1, :]  # [batch_size, 1, 1]
            embedded_i = self.feature_embedding(feature_i)  # [batch_size, 1, d_model]
            embedded_features.append(embedded_i)
        
        # 拼接所有特征嵌入
        x = torch.cat(embedded_features, dim=1)  # [batch_size, num_features, d_model]
        
        # 2. 权重增强（如果启用且提供了权重）
        if self.use_weight_enhancement and weights is not None:
            x = self._apply_weight_enhancement(x, weights)
        
        # 3. 添加CLS token（如果启用）
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)  # [batch_size, num_features+1, d_model]
        
        # 4. 位置编码
        x = self.positional_encoding(x)
        
        # 5. Transformer编码器
        for layer in self.transformer_layers:
            x = layer(x, mask)
        
        # 6. 提取表示用于分类
        if self.use_cls_token:
            # 使用CLS token的表示
            cls_representation = x[:, 0, :]  # [batch_size, d_model]
        else:
            # 使用平均池化
            cls_representation = x.mean(dim=1)  # [batch_size, d_model]
        
        # 7. 分类
        output = self.classification_head(cls_representation)
        
        return output
    
    def _apply_weight_enhancement(
        self, 
        x: torch.Tensor, 
        weights: torch.Tensor
    ) -> torch.Tensor:
        """应用权重增强"""
        # 权重投影
        weights_expanded = weights.unsqueeze(-1)  # [batch_size, num_features, 1]
        weight_embeddings = self.weight_projection(weights_expanded)  # [batch_size, num_features, d_model]
        
        # 权重融合
        enhanced_x = self.weight_fusion(x, weight_embeddings)
        
        return enhanced_x
    
    def compute_loss(
        self, 
        outputs: torch.Tensor, 
        targets: torch.Tensor,
        epoch: Optional[int] = None
    ) -> torch.Tensor:
        """计算损失"""
        return self.criterion(outputs, targets, epoch)
    
    def get_attention_weights(self, layer_idx: int = -1) -> torch.Tensor:
        """获取注意力权重（用于可视化）"""
        # 这里需要修改Transformer层以返回注意力权重
        # 简化实现，实际使用时需要在forward过程中保存注意力权重
        pass


class WeightFusionLayer(nn.Module):
    """权重融合层"""
    
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        
        # 融合门控机制
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        
        # 融合变换
        self.fusion_transform = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(
        self, 
        feature_embeddings: torch.Tensor, 
        weight_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        融合特征嵌入和权重嵌入
        
        Args:
            feature_embeddings: [batch_size, num_features, d_model]
            weight_embeddings: [batch_size, num_features, d_model]
        
        Returns:
            融合后的特征表示
        """
        # 拼接特征和权重嵌入
        concatenated = torch.cat([feature_embeddings, weight_embeddings], dim=-1)
        
        # 计算门控权重
        gate_weights = self.gate(concatenated)
        
        # 融合变换
        fused_features = self.fusion_transform(concatenated)
        
        # 加权融合
        enhanced_features = gate_weights * fused_features + (1 - gate_weights) * feature_embeddings
        
        # 残差连接和层归一化
        output = self.layer_norm(enhanced_features + feature_embeddings)
        
        return output


class ClassificationHead(nn.Module):
    """分类头"""
    
    def __init__(
        self,
        d_model: int,
        num_classes: int,
        dropout: float = 0.1,
        activation: str = 'relu',
        normalization: str = 'batch_norm'
    ):
        super().__init__()
        
        # 中间层
        self.hidden = nn.Linear(d_model, d_model // 2)
        
        # 激活函数
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'swish':
            self.activation = nn.SiLU()
        else:
            self.activation = nn.ReLU()
        
        # 归一化
        if normalization == 'batch_norm':
            self.norm = nn.BatchNorm1d(d_model // 2)
        elif normalization == 'layer_norm':
            self.norm = nn.LayerNorm(d_model // 2)
        else:
            self.norm = nn.Identity()
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 输出层
        self.output = nn.Linear(d_model // 2, num_classes)
        
        # 权重初始化
        self._init_weights()
    
    def _init_weights(self):
        """权重初始化"""
        nn.init.xavier_uniform_(self.hidden.weight)
        nn.init.constant_(self.hidden.bias, 0)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.constant_(self.output.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.hidden(x)
        x = self.activation(x)
        x = self.norm(x)
        x = self.dropout(x)
        x = self.output(x)
        return x


class FTTransformerModel(nn.Module):
    """基础Feature-based Transformer模型（无权重增强）"""
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        num_features: int = 10,
        num_classes: int = 2
    ):
        super().__init__()
        
        # 使用WE-FTT模型，但禁用权重增强
        self.model = WEFTTransformerModel(
            config=config,
            num_features=num_features,
            num_classes=num_classes,
            use_weight_enhancement=False
        )
    
    def forward(self, features: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.model(features, weights=None, mask=mask)
    
    def compute_loss(self, outputs: torch.Tensor, targets: torch.Tensor, epoch: Optional[int] = None) -> torch.Tensor:
        return self.model.compute_loss(outputs, targets, epoch)


def create_we_ftt_model(
    num_features: int = 10,
    num_classes: int = 2,
    config: Optional[Dict[str, Any]] = None,
    use_weight_enhancement: bool = True
) -> WEFTTransformerModel:
    """
    创建WE-FTT模型的便捷函数
    
    Args:
        num_features: 输入特征数量
        num_classes: 分类类别数量
        config: 模型配置参数
        use_weight_enhancement: 是否使用权重增强
    
    Returns:
        WE-FTT模型实例
    """
    if config is None:
        config = WEFTTConfig.BEST_PARAMS
    
    model = WEFTTransformerModel(
        config=config,
        num_features=num_features,
        num_classes=num_classes,
        use_weight_enhancement=use_weight_enhancement
    )
    
    return model


def create_ft_transformer_model(
    num_features: int = 10,
    num_classes: int = 2,
    config: Optional[Dict[str, Any]] = None
) -> FTTransformerModel:
    """
    创建基础FT-Transformer模型的便捷函数
    
    Args:
        num_features: 输入特征数量
        num_classes: 分类类别数量
        config: 模型配置参数
    
    Returns:
        FT-Transformer模型实例
    """
    model = FTTransformerModel(
        config=config,
        num_features=num_features,
        num_classes=num_classes
    )
    
    return model