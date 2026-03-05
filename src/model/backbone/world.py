import torch
import torch.nn as nn
import copy
from src.model.backbone.attention import LatentAttention
from src.model.backbone.rms import RMSNorm
from src.model.backbone.moe import SelfAdaptiveMoE

class SpatioTemporalBlock(nn.Module):
    """
    [演进阶段 6] 时空交织注意力块 (Spatio-Temporal Interleaved Block)
    结构: Spatial Attn -> Temporal Attn -> Shared FFN
    特性:
    1. 空间层: 双向注意力 (is_causal=False)，捕捉全局空间语义。
    2. 时间层: 单向注意力 (is_causal=True)，严格遵守因果律。
    3. Reshape 大法: 在 (B*T, S, C) 和 (B*S, T, C) 之间高效切换。
    4. 共享 FFN: 空间和时间特征共享同一个 MoE 专家组，存储通用物理规律。
    """
    def __init__(self, config):
        super().__init__()
        # 1. 空间配置 (双向)
        self.config_spatial = copy.deepcopy(config)
        self.config_spatial.is_causal = False
        
        # 2. 时间配置 (单向)
        self.config_temporal = copy.deepcopy(config)
        self.config_temporal.is_causal = True
        
        self.norm1 = RMSNorm(config.hidden_dim)
        self.spatial_attn = LatentAttention(self.config_spatial)
        
        self.norm2 = RMSNorm(config.hidden_dim)
        self.temporal_attn = LatentAttention(self.config_temporal)
        
        self.norm3 = RMSNorm(config.hidden_dim)
        self.mlp = SelfAdaptiveMoE(config) # Shared FFN
        
    def forward(self, x, past_key_values=None, use_cache=False):
        # x shape: (B, T, S, C)
        B, T, S, C = x.shape
        
        # 1. Spatial Attention: (B, T, S, C) -> (B*T, S, C)
        x_spatial = self.norm1(x).view(B*T, S, C)
        x_spatial = self.spatial_attn(x_spatial) 
        x = x + x_spatial.view(B, T, S, C)
        
        # 2. Temporal Attention: (B, T, S, C) -> (B*S, T, C)
        x_temporal_input = self.norm2(x).permute(0, 2, 1, 3).reshape(B*S, T, C)
        
        if use_cache:
            x_temporal, present_kv = self.temporal_attn(
                x_temporal_input, 
                past_key_value=past_key_values, 
                use_cache=True
            )
        else:
            if T > 1:
                x_temporal = self.temporal_attn(x_temporal_input)
            else:
                x_temporal = x_temporal_input
            present_kv = None
        
        x = x + x_temporal.view(B, S, T, C).permute(0, 2, 1, 3)
            
        # 3. Shared FFN: (B, T, S, C) -> (B*T, S, C)
        x_ffn = self.norm3(x).view(B*T, S, C) 
        x_ffn = self.mlp(x_ffn)
        x = x + x_ffn.view(B, T, S, C)
        
        if use_cache:
            return x, present_kv
        return x