import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.components.norm import RMSNorm2d

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
