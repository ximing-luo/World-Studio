import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    """
    RMSNorm (Root Mean Square Layer Normalization)
    比 LayerNorm 更快且效果相当，常用于现代 LLM (Llama, DeepSeek)
    """
    def __init__(self, dim: int, eps: float = 1e-4):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        # 强制转为 float32 计算以保证数值稳定性
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).type_as(x)

    def forward(self, x):
        return self.weight * self._norm(x)

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
