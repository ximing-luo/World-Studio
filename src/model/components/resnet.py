import torch
import torch.nn as nn

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

    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.SiLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, groups=out_channels, padding=1, bias=False),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0, bias=False),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.SiLU(inplace=True),

            nn.Conv2d(out_channels, out_channels* BottleNeck.expansion, kernel_size=1, bias=False),
            nn.GroupNorm(min(8, out_channels * BottleNeck.expansion), out_channels * BottleNeck.expansion)
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.GroupNorm(min(8, out_channels * BottleNeck.expansion), out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.SiLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class ResBlock(nn.Module):
    """
    通用残差块 (ResBlock)
    统一了 ResNet (压缩型) 和 MobileNetV2 (扩张型/倒残差) 的逻辑。
    
    设计要点:
    1. 计算中心化: 关注 mid_channels，即中间 3x3 深度卷积的计算维度。
    2. 架构统一: in -> mid (1x1) -> mid (3x3 DW) -> out (1x1)。
    3. LLM 风格: Pre-Norm (RMSNorm) + Linear Bottleneck (末尾无激活)。
    4. ResNet-D 优化: Shortcut 路径使用 AvgPool 进行平滑降维。
    """
    expansion = 4 # 默认 ResNet 风格
    
    def __init__(self, in_channels, out_channels, mid_channels=None, stride=1):
        super().__init__()
        
        if mid_channels is None:
            mid_channels = out_channels // 4 # 默认压缩 4 倍
            
        self.residual_function = nn.Sequential(
            # 1. 投影层: in -> mid (SiLU)
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.SiLU(inplace=True),
            # 中继归一化：压制由于通道扩张导致的内部数值膨胀
            nn.GroupNorm(min(8, mid_channels), mid_channels),
            
            # 2. 空间层: mid -> mid (3x3 Depthwise, SiLU)
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, groups=out_channels, padding=1, bias=False),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=1, padding=0, bias=False),
            nn.GroupNorm(min(8, mid_channels), mid_channels),
            nn.SiLU(inplace=True),
            
            # 3. 输出层: mid -> out (无激活, 线性瓶颈)
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(min(8, out_channels), out_channels)
        )
        
        # 残差分支输出归一化 (Post-Norm)
        # 确保残差路径的输出量纲也被物理锁定在 1.0 附近
        self.res_norm = RMSNorm2d(out_channels)
        
        # Shortcut 对齐 (ResNet-D 优化版)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True) if stride > 1 else nn.Identity(),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                RMSNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        # 核心改进：双路径输出归一化
        # 1. 计算残差分支并立即归一化
        out = self.res_norm(self.residual_function(x))
        # 2. 与同样归一化过的捷径分支相加 (1.0 + 1.0 ≈ 2.0)
        return out + identity


    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.SiLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, groups=out_channels, padding=1, bias=False),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0, bias=False),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.SiLU(inplace=True),

            nn.Conv2d(out_channels, out_channels* BottleNeck.expansion, kernel_size=1, bias=False),
            nn.GroupNorm(min(8, out_channels * BottleNeck.expansion), out_channels * BottleNeck.expansion)
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.GroupNorm(min(8, out_channels * BottleNeck.expansion), out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.SiLU(inplace=True)(self.residual_function(x) + self.shortcut(x))
