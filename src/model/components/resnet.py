import torch
import torch.nn as nn
from src.model.ecr.ecr import EfficientEvolutionLayer, CrossScholarFusion
from src.model.backbone.rms import RMSNorm2d

class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, groups=in_channels, padding=1, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, groups=out_channels, padding=1, bias=False),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=1, padding=0, bias=False),
            nn.GroupNorm(min(8, out_channels * BasicBlock.expansion), out_channels * BasicBlock.expansion)
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(min(8, out_channels * BasicBlock.expansion), out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.SiLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class BottleNeck(nn.Module):
    """
    反向瓶颈块 (Inverted Bottleneck)
    相比传统 BottleNeck (宽->窄->宽)，采用 (窄->宽->窄) 结构。
    1. 1x1 扩张层: 增加通道数，在高维空间提取特征。
    2. 3x3 深度卷积: 极低算力消耗下进行空间特征聚合。
    3. 1x1 投影层: 将特征压回输出维度，减少后续显存占用。
    """
    expansion = 2
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        mid_channels = in_channels * self.expansion

        self.residual_function = nn.Sequential(
            # 1. 扩张层: in -> mid
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.GroupNorm(min(8, mid_channels), mid_channels),
            nn.SiLU(inplace=True),

            # 2. 深度卷积层: mid -> mid (DW)
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, groups=mid_channels, padding=1, bias=False),
            nn.GroupNorm(min(8, mid_channels), mid_channels),
            nn.SiLU(inplace=True),

            # 3. 投影层: mid -> out * expansion
            nn.Conv2d(mid_channels, out_channels * self.expansion, kernel_size=1, bias=False),
            nn.GroupNorm(min(8, out_channels * self.expansion), out_channels * self.expansion)
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, stride=stride, kernel_size=1, bias=False),
                nn.GroupNorm(min(8, out_channels * self.expansion), out_channels * self.expansion)
            )

    def forward(self, x):
        return nn.SiLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class ResBlock(nn.Module):
    """
    大脑化残差块 (Brain-inspired ResBlock)
    参考 ConvNeXt / MobileNetV3 的 Inverted Bottleneck 结构。
    
    设计哲学:
    1. 高参数、低激活: 通过 1x1 卷积将通道扩张 4 倍，存储海量特征模式。
    3. 线性瓶颈: 末尾不使用激活函数，保持信息流的完整性。
    4. 归一化策略: 使用 RMSNorm2d 进行稳定归一化。
    """
    def __init__(self, in_channels, out_channels, mid_channels=None, stride=1):
        super().__init__()
        
        if mid_channels is None:
            mid_channels = out_channels * 2 # 默认放大 4 倍
            
        self.residual_function = nn.Sequential(
            # 1. 投影层: in -> mid (SiLU)
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, groups=in_channels, bias=False),
            nn.SiLU(inplace=True),
            # 中继归一化：压制由于通道扩张导致的内部数值膨胀
            nn.GroupNorm(min(8, mid_channels), mid_channels),
            
            # 2. 空间层: mid -> mid (3x3 Depthwise, SiLU)
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, groups=mid_channels, padding=1, bias=False),
            nn.GroupNorm(min(8, mid_channels), mid_channels),
            nn.SiLU(inplace=True),
            
            # 3. 输出层: mid -> out (无激活, 线性瓶颈)
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(min(8, out_channels), out_channels)
        )
        
        # Shortcut 对齐 (ResNet-D 优化版)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True) if stride > 1 else nn.Identity(),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.GroupNorm(min(8, out_channels), out_channels)
            )

    def forward(self, x):
        return nn.SiLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class EResBlock(nn.Module):
    """
    大脑化残差块 (Brain-inspired ResBlock)
    参考 ConvNeXt / MobileNetV3 的 Inverted Bottleneck 结构。
    
    设计哲学:
    1. 高参数、低激活: 通过 1x1 卷积将通道扩张 4 倍，存储海量特征模式。
    3. 线性瓶颈: 末尾不使用激活函数，保持信息流的完整性。
    4. 归一化策略: 使用 RMSNorm2d 进行稳定归一化。
    """
    def __init__(self, in_channels, out_channels, mid_channels=None, stride=1, num_evolve_layers=3):
        super().__init__()
        
        if mid_channels is None:
            mid_channels = out_channels * 2 # 默认放大 4 倍
            
        self.residual_function = nn.Sequential(
            # 1. 投影层: in -> mid (SiLU)
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.SiLU(inplace=True),
            # 中继归一化：压制由于通道扩张导致的内部数值膨胀
            nn.GroupNorm(min(8, mid_channels), mid_channels),
            
            # 2. 空间层: mid -> mid (3x3 Depthwise, SiLU)
            nn.Sequential(*[
            EfficientEvolutionLayer(mid_channels, kernel_size=3) for _ in range(num_evolve_layers)
        ]),
            
            # 3. 输出层: mid -> out (无激活, 线性瓶颈)
            CrossScholarFusion(mid_channels, out_channels),
            nn.GroupNorm(min(8, out_channels), out_channels)
        )
        
        # Shortcut 对齐 (ResNet-D 优化版)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True) if stride > 1 else nn.Identity(),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.GroupNorm(min(8, out_channels), out_channels)
            )

    def forward(self, x):
        return nn.SiLU(inplace=True)(self.residual_function(x) + self.shortcut(x))
