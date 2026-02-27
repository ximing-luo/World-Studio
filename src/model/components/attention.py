import torch
import torch.nn as nn
import torch.nn.functional as F

from .rms import RMSNorm

class GatedMLP(nn.Module):
    """
    Gated MLP (SwiGLU 变体) - 借鉴自 LLM 的高效特征蒸馏模块。
    结构: Down(SiLU(Gate(LN(x))) * Up(LN(x)))
    """
    def __init__(self, input_dim, output_dim, intermediate_size=None, bias=True):
        super().__init__()
        if intermediate_size is None:
            # 默认 8/3 倍扩展，并对齐到 64 的倍数
            intermediate_size = int(output_dim * 8 / 3)
            intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)
            
        self.norm = RMSNorm(input_dim)
        self.gate = nn.Linear(input_dim, intermediate_size, bias=False)
        self.up_proj = nn.Linear(input_dim, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, output_dim, bias=bias)
        self.act_func = F.silu

    def forward(self, x):
        # 先进行归一化，防止深层数值爆炸
        x = self.norm(x)
        # Down(SiLU(Gate(x)) * Up(x))
        return self.down_proj(self.act_func(self.gate(x)) * self.up_proj(x))

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block (通道注意力机制)
    通过全局平均池化捕捉通道间的全局统计信息，学习通道重要性权重。
    """
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)