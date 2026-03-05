import torch
import torch.nn as nn
import torch.nn.functional as F

from .rms import RMSNorm, RMSNorm2d

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

class Focus(nn.Module):
    """
    Focus (下采样) 模块 - 标准功能块实现。
    包含 PixelUnshuffle + 1x1 卷积 + 归一化 + 激活。
    逻辑: 将空间信息无损压缩至通道，再通过 1x1 卷积进行特征整合。
    """
    def __init__(self, in_channels, out_channels, block_size=2, act=True):
        super().__init__()
        self.block_size = block_size
        # 1x1 卷积用于整合切片后的通道特征
        self.conv = nn.Conv2d(in_channels * (block_size ** 2), out_channels, 1, bias=False)
        self.norm = RMSNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        # PixelUnshuffle 将 (B, C, H, W) -> (B, C*bs^2, H/bs, W/bs)
        x = F.pixel_unshuffle(x, self.block_size)
        return self.act(self.norm(self.conv(x)))

class UnFocus(nn.Module):
    """
    UnFocus (上采样) 模块 - 标准功能块实现。
    包含 1x1 卷积 + 归一化 + 激活 + PixelShuffle。
    逻辑: 先通过卷积准备足够的通道数，再通过 PixelShuffle 还原到空间维度。
    """
    def __init__(self, in_channels, out_channels, block_size=2, act=True):
        super().__init__()
        self.block_size = block_size
        # 先卷积到 pixel_shuffle 所需的通道数 (out_channels * bs^2)
        self.conv = nn.Conv2d(in_channels, out_channels * (block_size ** 2), 1, bias=False)
        self.norm = RMSNorm2d(out_channels * (block_size ** 2))
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        x = self.act(self.norm(self.conv(x)))
        return F.pixel_shuffle(x, self.block_size)

class AttentionPooling(nn.Module):
    """
    注意力池化 (Attention Pooling)
    使用一个可学习的 Query 向量从空间 Tokens 中提取全局语义摘要。
    相比简单的 Flatten + Linear，它具有空间平移不变性，且能聚焦于画面中的核心实体。
    """
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim))
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # x: (B, C, H, W) -> (B, S, C) 其中 S = H * W
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        
        # 扩展 Query 到 Batch Size
        q = self.query.expand(B, -1, -1)
        
        # Cross-Attention: 用一个 Query 去问所有的空间 Tokens
        # attn_out: (B, 1, C)
        attn_out, _ = self.attn(q, x, x)
        
        # 归一化并压缩维度 -> (B, C)
        return self.norm(attn_out).squeeze(1)


