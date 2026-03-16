import torch
import torch.nn as nn
from .ops_norm import rms_norm_2d, layer_norm_2d

class RMSNorm2d(nn.Module):
    """
    RMSNorm2d: 专门针对视觉张量 (B, C, H, W) 的归一化。
    在通道维 (dim=1) 上进行归一化，支持直接广播。
    已优化为 CUDA 融合算子，减少 kernel launch 开销。
    """
    def __init__(self, channels: int, eps: float = 1e-4):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))

    def forward(self, x):
        return rms_norm_2d(x, self.weight, self.eps)

class LayerNorm2d(nn.Module):
    """
    LayerNorm2d: 专门针对视觉张量 (B, C, H, W) 的归一化。
    在通道维 (dim=1) 上进行归一化，包含均值中心化和方差缩放。
    已优化为 CUDA 融合算子，大幅提升效率。
    """
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # 仿射变换参数：缩放和偏移
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        return layer_norm_2d(x, self.weight, self.bias, self.eps)
