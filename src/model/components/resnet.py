import torch
import torch.nn as nn
from src.model.ecr.norm import RMSNorm2d, LayerNorm2d

class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.residual_function = nn.Sequential(
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, groups=in_channels, padding=1, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False),
            RMSNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, groups=out_channels, padding=1, bias=False),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0, bias=False),
            RMSNorm2d(out_channels)
        )

        # Shortcut 对齐 (ResNet-D 优化版)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True) if stride > 1 else nn.Identity(),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            )

        self.norm = LayerNorm2d(in_channels) if stride != 1 else nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        return self.shortcut(x) + self.residual_function(x)

class BottleNeck(nn.Module):
    """
    反向瓶颈块 (Inverted Bottleneck)
    相比传统 BottleNeck (宽->窄->宽)，采用 (窄->宽->窄) 结构。
    1. 1x1 扩张层: 增加通道数，在高维空间提取特征。
    2. 3x3 深度卷积: 极低算力消耗下进行空间特征聚合。
    3. 1x1 投影层: 将特征压回输出维度，减少后续显存占用。
    """
    def __init__(self, in_channels, out_channels, mid_channels=None, stride=1):
        super().__init__()
        if mid_channels is None:
            mid_channels = in_channels * 4

        self.residual_function = nn.Sequential(
            # 1. 扩张层: in -> mid
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.SiLU(inplace=True),

            # 2. 深度卷积层: mid -> mid (DW)
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, groups=mid_channels, padding=1, bias=False),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=1, bias=False),
            RMSNorm2d(mid_channels),

            # 3. 投影层: mid -> out * expansion
            nn.SiLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        )

        # Shortcut 对齐 (ResNet-D 优化版)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True) if stride > 1 else nn.Identity(),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            )
        
        self.norm = LayerNorm2d(in_channels) if stride != 1 else nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        return self.shortcut(x) + self.residual_function(x)
