import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block (通道注意力机制)
    通过全局平均池化捕捉通道间的全局统计信息，学习通道重要性权重。
    """
    def __init__(self, channels, reduction=4):
        super().__init__()
        mid_channels = max(32, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid_channels, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(mid_channels, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

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


