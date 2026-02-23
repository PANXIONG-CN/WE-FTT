"""
Model components for WE-FTT project.

This module contains reusable neural network components such as attention layers,
loss functions, and learning rate schedulers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, Tuple


class MultiHeadedAttention(nn.Module):
    """多头注意力机制层"""
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        kv_compression_ratio: Optional[float] = None,
        kv_compression_sharing: Optional[str] = None
    ):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.dropout = dropout
        
        # 线性变换层
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        # KV压缩（如果启用）
        self.kv_compression_ratio = kv_compression_ratio
        self.kv_compression_sharing = kv_compression_sharing
        
        if kv_compression_ratio is not None:
            compressed_d_k = int(self.d_k * kv_compression_ratio)
            self.k_compression = nn.Linear(self.d_k, compressed_d_k)
            self.v_compression = nn.Linear(self.d_k, compressed_d_k)
            self.d_k_compressed = compressed_d_k
        
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size = query.size(0)
        
        # 1) 线性变换并重塑为多头
        Q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # 2) KV压缩（如果启用）
        if self.kv_compression_ratio is not None:
            K = self.k_compression(K)
            V = self.v_compression(V)
            d_k = self.d_k_compressed
        else:
            d_k = self.d_k
        
        # 3) 计算注意力
        attn_output = self._scaled_dot_product_attention(Q, K, V, mask, d_k)
        
        # 4) 连接多头并通过最终线性层
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.n_heads * self.d_k
        )
        
        if self.kv_compression_ratio is not None:
            # 如果使用了KV压缩，需要调整输出维度
            attn_output = self.w_o(attn_output)
        else:
            attn_output = self.w_o(attn_output)
        
        return attn_output
    
    def _scaled_dot_product_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor, 
        V: torch.Tensor,
        mask: Optional[torch.Tensor],
        d_k: int
    ) -> torch.Tensor:
        """缩放点积注意力"""
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        
        # 应用掩码（如果有）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout_layer(attn_weights)
        
        # 加权求和
        attn_output = torch.matmul(attn_weights, V)
        
        return attn_output


class DynamicFocalLoss(nn.Module):
    """动态焦点损失函数"""
    
    def __init__(
        self,
        alpha: float = 1.0,
        gamma: float = 2.0,
        reduction: str = 'mean',
        adaptive_gamma: bool = True
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.adaptive_gamma = adaptive_gamma
        
    def forward(
        self, 
        inputs: torch.Tensor, 
        targets: torch.Tensor,
        epoch: Optional[int] = None
    ) -> torch.Tensor:
        # 计算交叉熵损失
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # 计算概率
        pt = torch.exp(-ce_loss)
        
        # 动态调整gamma
        if self.adaptive_gamma and epoch is not None:
            # 在训练早期使用较小的gamma，后期增大
            dynamic_gamma = self.gamma * (1 + 0.1 * epoch)
        else:
            dynamic_gamma = self.gamma
        
        # 计算焦点损失
        focal_loss = self.alpha * (1 - pt) ** dynamic_gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class WarmupCosineSchedule(torch.optim.lr_scheduler._LRScheduler):
    """带预热的余弦退火学习率调度器"""
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        t_total: int,
        cycles: float = 0.5,
        last_epoch: int = -1
    ):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # 预热阶段：线性增长
            return [
                base_lr * float(self.last_epoch) / float(max(1, self.warmup_steps))
                for base_lr in self.base_lrs
            ]
        else:
            # 余弦退火阶段
            progress = float(self.last_epoch - self.warmup_steps) / float(
                max(1, self.t_total - self.warmup_steps)
            )
            return [
                base_lr * max(0.0, 0.5 * (1.0 + math.cos(math.pi * self.cycles * 2.0 * progress)))
                for base_lr in self.base_lrs
            ]


class PositionalEncoding(nn.Module):
    """正弦余弦位置编码"""
    
    def __init__(self, d_model: int, max_len: int = 5000, learnable: bool = False):
        super().__init__()
        self.d_model = d_model
        self.learnable = learnable
        
        if learnable:
            # 可学习的位置编码
            self.pos_encoding = nn.Parameter(torch.randn(max_len, d_model))
        else:
            # 固定的正弦余弦位置编码
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            
            div_term = torch.exp(
                torch.arange(0, d_model, 2).float() * 
                (-math.log(10000.0) / d_model)
            )
            
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            
            self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.learnable:
            seq_len = x.size(1)
            return x + self.pos_encoding[:seq_len, :].unsqueeze(0)
        else:
            seq_len = x.size(1)
            return x + self.pe[:seq_len, :].unsqueeze(0)


class FeatureEmbedding(nn.Module):
    """特征嵌入层"""
    
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        dropout: float = 0.1,
        use_bias: bool = True
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        
        # 线性变换
        self.linear = nn.Linear(input_dim, d_model, bias=use_bias)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 权重初始化
        self._init_weights()
    
    def _init_weights(self):
        """权重初始化"""
        nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.constant_(self.linear.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 线性变换
        x = self.linear(x)
        
        # Layer normalization
        x = self.layer_norm(x)
        
        # Dropout
        x = self.dropout(x)
        
        return x


class FeedForward(nn.Module):
    """前馈神经网络层"""
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = 'relu'
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        
        # 两层全连接网络
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        
        # 激活函数
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'swish':
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 权重初始化
        self._init_weights()
    
    def _init_weights(self):
        """权重初始化"""
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.constant_(self.linear1.bias, 0)
        nn.init.constant_(self.linear2.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer编码器块"""
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = 'relu',
        prenormalization: bool = True,
        kv_compression_ratio: Optional[float] = None,
        kv_compression_sharing: Optional[str] = None
    ):
        super().__init__()
        self.prenormalization = prenormalization
        
        # 多头注意力
        self.self_attention = MultiHeadedAttention(
            d_model, n_heads, dropout, kv_compression_ratio, kv_compression_sharing
        )
        
        # 前馈网络
        self.feed_forward = FeedForward(d_model, d_ff, dropout, activation)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 自注意力子层
        if self.prenormalization:
            # Pre-normalization
            normed_x = self.norm1(x)
            attn_output = self.self_attention(normed_x, normed_x, normed_x, mask)
            x = x + self.dropout1(attn_output)
            
            # 前馈子层
            normed_x = self.norm2(x)
            ff_output = self.feed_forward(normed_x)
            x = x + self.dropout2(ff_output)
        else:
            # Post-normalization
            attn_output = self.self_attention(x, x, x, mask)
            x = self.norm1(x + self.dropout1(attn_output))
            
            ff_output = self.feed_forward(x)
            x = self.norm2(x + self.dropout2(ff_output))
        
        return x


def get_sinusoidal_encoding_table(n_position: int, d_hid: int) -> torch.Tensor:
    """生成正弦余弦位置编码表"""
    
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]
    
    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    
    return torch.FloatTensor(sinusoid_table).unsqueeze(0)