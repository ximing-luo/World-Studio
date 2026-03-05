import torch
import torch.nn as nn
from types import SimpleNamespace
from configs.world import VisionThinkingConfig
from src.model.backbone.transform import DeepSeekV3Block

class BaseVision(nn.Module):
    """
    视觉模块基类，定义标准的感知与重构接口。
    """
    def __init__(self):
        super().__init__()

    def encode(self, x):
        """将图像/视频编码为隐空间表示。"""
        raise NotImplementedError

    def decode(self, z):
        """从隐空间表示还原图像/视频。"""
        raise NotImplementedError

    def forward(self, x):
        """默认行为：重构。"""
        z = self.encode(x)
        # 如果是 VAE (返回 mu, logvar)，取 mu 重构
        if isinstance(z, tuple):
            z = z[0]
        return self.decode(z)

class ThinkingSpace(nn.Module):
    """
    思维空间 (Thinking Space) - 视觉感知的语义增强器。
    设计哲学:
    1. 纯粹空间: 专注于梳理当前帧的语义，不涉及时间预测，因此不需要 KV 缓存。
    2. 架构对齐: 使用 DeepSeekV3Block (MLA + MoE) 实现高效的全局特征建模。
    """
    def __init__(self, channels, height, width, num_layers=8, num_experts=8):
        super().__init__()
        self.channels = channels
        self.height = height
        self.width = width
        self.hidden_dim = channels 
        
        # 内部管理配置 (使用专注于语义理解的配置)
        self.config = VisionThinkingConfig(
            hidden_dim=self.hidden_dim,
            num_experts=num_experts,
            n_layer=num_layers
        )
        
        # 使用标准的 Transformer 块处理空间特征
        self.blocks = nn.ModuleList([DeepSeekV3Block(self.config) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(self.hidden_dim)

    def forward(self, x):
        """
        输入: (B, C, H, W) 或 (B, C, T, H, W)
        输出: 形状与输入一致，但特征已被语义增强。
        """
        # 1. 统一处理输入形状
        is_video = x.dim() == 5
        if is_video:
            b, c, t, h, w = x.shape
            # 将 (B, C, T, H, W) 展平为 (B*T, H*W, C)，独立处理每一帧的空间信息
            x = x.permute(0, 2, 3, 4, 1).reshape(b * t, h * w, c)
        else:
            b, c, h, w = x.shape
            # 将 (B, C, H, W) 展平为 (B, H*W, C)
            x = x.permute(0, 2, 3, 1).reshape(b, h * w, c)
            
        # 2. 空间特征建模 (Transformer 遍历)
        for block in self.blocks:
            x = block(x)
                
        x = self.norm(x)
        
        # 3. 还原回原始形状
        if is_video:
            x = x.view(b, t, h, w, c).permute(0, 4, 1, 2, 3)
        else:
            x = x.view(b, h, w, c).permute(0, 3, 1, 2)
            
        return x



