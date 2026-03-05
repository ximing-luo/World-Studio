import torch
import torch.nn as nn
from configs.world import PredictorConfig
from src.model.backbone.transform import DeepSeekV3Block
from src.model.backbone.world import SpatioTemporalBlock

class BasePredictor(nn.Module):
    """
    预测器基类，定义标准接口。
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, x):
        raise NotImplementedError

class MLPPredictor(BasePredictor):
    """
    全连接映射预测器。
    用于简单的空间信息分析或线性投影。
    """
    def __init__(self, input_dim, output_dim, hidden_dim=None):
        super().__init__(input_dim, output_dim)
        hidden_dim = hidden_dim if hidden_dim is not None else input_dim * 2
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class TransformerPredictor(BasePredictor):
    """
    基于 Transformer (DeepSeekV3Block) 的预测器。
    用于复杂的时序推演或全局空间关联分析。
    """
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers=8, num_heads=8):
        super().__init__(input_dim, output_dim)
        
        # 内部管理配置 (使用统一的世界模型配置)
        self.config = PredictorConfig(
            hidden_dim=hidden_dim,
            n_layer=num_layers,
            n_head=num_heads,
            n_kv_head=2,
            max_seq_len=1024
        )
        
        # 维度映射：Input -> Transformer Hidden Dim
        self.input_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        
        # Transformer Blocks
        self.blocks = nn.ModuleList([
            DeepSeekV3Block(self.config) for _ in range(num_layers)
        ])
        
        # 维度映射：Transformer Hidden Dim -> Output
        self.output_head = nn.Linear(hidden_dim, output_dim) if output_dim != hidden_dim else nn.Identity()

    def forward(self, x):
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        return self.output_head(h)

class SpatioTemporalPredictor(BasePredictor):
    """
    时空融合版预测器 (Spatio-Temporal Predictor) - 世界模型的核心组件。
    设计哲学:
    1. 简化接口: 只暴露业务参数 (input/output/hidden/size)，隐藏模型实现细节。
    2. 继承基类: 符合预测器标准接口，可无缝插拔到 JEPA/RSSM。
    """
    def __init__(self, input_dim, output_dim, hidden_dim, height, width, num_layers=8):
        super().__init__(input_dim, output_dim)
        self.height = height
        self.width = width
        self.hidden_dim = hidden_dim
        
        # 内部管理配置 (使用统一的世界模型配置)
        self.config = PredictorConfig(
            hidden_dim=hidden_dim
        )
        
        # 维度投影
        self.input_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        
        # 时空块堆叠
        self.blocks = nn.ModuleList([SpatioTemporalBlock(self.config) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(hidden_dim)
        
        # 输出投影
        self.output_head = nn.Linear(hidden_dim, output_dim) if output_dim != hidden_dim else nn.Identity()
        
        # KV Cache 状态
        self.past_key_values = None

    def reset_state(self):
        """重置序列状态"""
        self.past_key_values = None

    def forward_with_cache(self, x):
        """流式推理"""
        output, present_kv = self.forward(x, past_key_values=self.past_key_values, use_cache=True)
        self.past_key_values = present_kv
        return output

    def forward(self, x, past_key_values=None, use_cache=False):
        """
        支持 (B, L, D) 或 (B, T, S, D) 输入。
        """
        if x.dim() == 3:
            B, L, D = x.shape
            S = self.height * self.width
            T = L // S
            x = x.view(B, T, S, D)
        elif x.dim() == 4:
            B, T, S, D = x.shape
        else:
            raise ValueError(f"Unsupported input shape: {x.shape}")
            
        x = self.input_proj(x) 
        
        present_key_values = [] if use_cache else None
        
        for i, block in enumerate(self.blocks):
            if use_cache:
                layer_past = past_key_values[i] if past_key_values is not None else None
                x, layer_present = block(x, past_key_values=layer_past, use_cache=True)
                present_key_values.append(layer_present)
            else:
                x = block(x)
        
        x = self.norm(x)
        x = self.output_head(x)
        
        if x.dim() == 4:
            B, T, S, D = x.shape
            x = x.view(B, T*S, D)
            
        if use_cache:
            return x, present_key_values
        return x

