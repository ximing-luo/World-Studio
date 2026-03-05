import torch
from torch import nn
from torch.nn import functional as F
import copy
from src.model.components.rms import RMSNorm2d
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
        
    def forward(self, x):
        # x shape: (B, T, S, C)
        B, T, S, C = x.shape
        
        # 1. Spatial Attention: (B, T, S, C) -> (B*T, S, C)
        # 空间混合：在每一帧内部，所有位置(S)互相可见
        shortcut = x
        x_spatial = self.norm1(x).view(B*T, S, C)
        x_spatial = self.spatial_attn(x_spatial) 
        x = x + x_spatial.view(B, T, S, C)
        
        # 2. Temporal Attention: (B, T, S, C) -> (B*S, T, C)
        # 时间混合：在每一个位置，跨帧(T)进行因果推理
        if T > 1:
            shortcut = x
            # 交换 T 和 S 维度
            x_temporal = self.norm2(x).permute(0, 2, 1, 3).reshape(B*S, T, C)
            x_temporal = self.temporal_attn(x_temporal)
            # 还原维度
            x = x + x_temporal.view(B, S, T, C).permute(0, 2, 1, 3)
            
        # 3. Shared FFN: (B, T, S, C) -> (B*T, S, C)
        # 物理规律应用：对每一个点(B*T*S)应用相同的物理法则
        shortcut = x
        x_ffn = self.norm3(x).view(B*T, S, C) # FFN 通常对 batch*seq 处理
        x_ffn = self.mlp(x_ffn)
        x = x + x_ffn.view(B, T, S, C)
        
        return x

class ThinkingSpace(nn.Module):
    """
    思维空间 (Thinking Space) v7.0 - 时空融合版 (Sora/VideoGPT Style)
    设计哲学:
    1. 高维语义: Hidden Dim = Channels (512)，不再是 H*W。
    2. 时空分离: 使用 SpatioTemporalBlock 分别处理空间和时间。
    3. 维度重塑: 支持 (B, C, H, W) 图像输入和 (B, C, T, H, W) 视频输入。
    """
    def __init__(self, channels, height, width, num_layers=16, num_experts=16):
        super().__init__()
        self.channels = channels
        self.height = height
        self.width = width
        # [重大变更] hidden_dim 现在等于 channels (512)，而不是 pixels (32)
        # 这意味着模型在 512 维的语义空间中思考，而不是在 32 维的像素空间中思考
        self.hidden_dim = channels 
        
        class Config:
            def __init__(self, hidden_dim, num_experts):
                self.hidden_dim = hidden_dim
                self.num_experts = num_experts
                self.num_experts_per_tok = 2
                self.num_shared_experts = 1
                self.kv_lora_rank = 16 # 可以适当增加以匹配 512 维
                self.q_lora_rank = 16
                self.qk_rope_head_dim = 64 # 增加 Head Dim
                self.qk_nope_head_dim = 64
                self.v_head_dim = 64
                self.n_head = 8 # 增加 Head 数
                self.n_kv_head = 2
                self.dropout = 0.1
                self.bias = False
                self.intermediate_size = hidden_dim * 2
                self.bias_update_rate = 0.001
                self.max_seq_len = 1024 # 足够覆盖 T=16, S=32
                self.rope_scale = 1.0
                self.rope_base = 10000.0
                self.router_aux_loss_coef = 0.001 # 如果用 SoftBalancedMoE
        
        config = Config(self.hidden_dim, num_experts)
        
        # 使用时空块堆叠
        self.blocks = nn.ModuleList([SpatioTemporalBlock(config) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(self.hidden_dim)

    def forward(self, x):
        # 支持两种输入模式:
        # 1. Image: (B, C, H, W) -> T=1
        # 2. Video: (B, C, T, H, W)
        
        is_video = x.dim() == 5
        if not is_video:
            b, c, h, w = x.shape
            t = 1
            x = x.unsqueeze(2) # (B, C, 1, H, W)
        else:
            b, c, t, h, w = x.shape
            
        # 转换为 (B, T, S, C) 格式
        # S = H * W
        x = x.permute(0, 2, 3, 4, 1).reshape(b, t, h*w, c)
        
        # 深度思考
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        # 还原格式
        x = x.reshape(b, t, h, w, c).permute(0, 4, 1, 2, 3) # (B, C, T, H, W)
        
        if not is_video:
            x = x.squeeze(2) # (B, C, H, W)
            
        return x
