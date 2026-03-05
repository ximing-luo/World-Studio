import torch
import torch.nn as nn

class RMSNorm2d(nn.Module):
    """
    RMSNorm2d: 专门针对视觉张量 (B, C, H, W) 的归一化。
    在通道维 (dim=1) 上进行归一化，支持直接广播。
    """
    def __init__(self, channels: int, eps: float = 1e-4):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))

    def _norm(self, x):
        # 计算通道维的均方根
        # x shape: (B, C, H, W) -> rms shape: (B, 1, H, W)
        return x * torch.rsqrt(x.float().pow(2).mean(1, keepdim=True) + self.eps).type_as(x)

    def forward(self, x):
        return self.weight * self._norm(x)

class LayerNorm2d(nn.Module):
    """
    LayerNorm2d: 专门针对视觉张量 (B, C, H, W) 的归一化。
    在通道维 (dim=1) 上进行归一化，包含均值中心化和方差缩放。
    相比 RMSNorm，它能更强力地抑制均值漂移和激活爆炸。
    """
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # 仿射变换参数：缩放和偏移
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        # 1. 计算通道维均值 (Mean Centering)
        # 这一步是抑制参数爆炸的关键，强制将特征向量中心化到零点
        mean = x.mean(1, keepdim=True)
        # 2. 计算通道维方差
        var = (x - mean).pow(2).mean(1, keepdim=True)
        # 3. 标准化
        x = (x - mean) * torch.rsqrt(var + self.eps)
        # 4. 仿射变换
        return x * self.weight + self.bias
