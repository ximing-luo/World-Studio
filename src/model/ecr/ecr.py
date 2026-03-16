import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.components.attention import SEBlock
from src.model.components.norm import RMSNorm2d, LayerNorm2d
from src.model.ecr.cuda_evolution.ops_evolution import EvolutionLayer

class CrossScholarFusion(nn.Module):
    """
    交叉学者融合 (Cross Scholar Fusion)
    适合高通道低分辨率
    
    设计哲学:
    1. 语义投影 (W_K): 大小 32 x C，代表 32 种预定义的“语义主题”。
    2. 偏好检索 (W_Q): 大小 C x 32，代表通道对这 32 种主题的响应权重。
    3. 核心逻辑: 利用低秩近似 (Low-rank Approximation) 实现类似交叉注意力的全局通道融合。
    4. 计算优化: 利用结合律 W_Q @ (W_K @ X)，将计算复杂度从 O(C^2) 降低到 O(C * L)，其中 L 为潜空间维度 (32)。
    """
    def __init__(self, in_channels, out_channels, latent_dim=None):
        super().__init__()
        if latent_dim is None:
            latent_dim = in_channels // 16
        if latent_dim < 64:
            latent_dim = 64
        self.latent_dim = latent_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # W_K: latent_dim x in_channels (归一化到学者空间)
        # 优化：预先转换为 1x1 卷积核形状
        w_k = torch.randn(latent_dim, in_channels, 1, 1) * (latent_dim ** -0.5)
        self.W_K = nn.Parameter(w_k)
        
        # W_Q: out_channels x latent_dim (从学者空间还原)
        w_q = torch.randn(out_channels, latent_dim, 1, 1) * (latent_dim ** -0.5)
        self.W_Q = nn.Parameter(w_q)

    def forward(self, x):
        # 1. 投影到潜空间 (分类): (B, latent_dim, H, W) = W_K @ X
        x = F.conv2d(x, self.W_K)
        
        # 2. 从潜空间检索并还原: (B, out_channels, H, W) = W_Q @ 潜空间
        x = F.conv2d(x, self.W_Q)
        
        return x

class EfficientEvolutionLayer(nn.Module):
    """
    高效演化层 (Efficient Evolution Layer)
    
    设计哲学:
    - 采用极致的分组卷积 (Depthwise Convolution)，消除通道间冗余计算。
    - 专注提取空间特征，保持通道间的独立性。
    """
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.conv = nn.Sequential(
            # 极致优化: 使用 ReLU 代替 LeakyReLU，保持 1-bit mask 级别的显存优势
            nn.ReLU(inplace=False), # 必须为 False，否则会破坏残差连接的输入 x
            nn.Conv2d(channels, channels, kernel_size, padding=kernel_size//2, groups=channels, bias=True)
        )
        
    def forward(self, x):
        return x + self.conv(x)

class EfficientCrossResBlock(nn.Module):
    """
    高效交叉残差块 (Efficient Cross Residual Block)
    
    设计哲学:
    - 针对传统残差块在通道融合中浪费约 98% 算力的痛点进行优化。
    - 空间特征提取: 通过极致的分组卷积 (EfficientEvolutionLayer) 进行“静默”演化，不进行通道间沟通。
    - 全局语义融合: 最后通过 CrossScholarFusion (基于低秩潜空间的交叉注意力机制) 实现高效的全局通道交互。
    - 架构演进: 将传统的 Dense 卷积残差结构转变为“稀疏演化 + 瓶颈融合”的高效范式。
    - 显存优化: 支持梯度检查点 (Gradient Checkpointing)，在大规模深度演化时可大幅降低训练显存。
    """
    def __init__(self, in_channels, out_channels, expansion=0.5, stride=1, num_evolve_layers=4):
        super().__init__()
        # mid_channels 为最终的输出通道数，由 expansion 决定
        mid_channels = int(in_channels * expansion)

        # 1. 分组膨胀与下采样 (Expansion & Downsample)
        # 用极低成本的分组卷积实现通道暴涨，物理隔离通道
        group = int(min(in_channels, mid_channels))
        self.expand = nn.Sequential(
            LayerNorm2d(in_channels),
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=stride, padding=1, groups=group, bias=False)
        )
        self.seblock = SEBlock(mid_channels, reduction=4)
        self.evolution = nn.Sequential(*[EvolutionLayer(mid_channels) for _ in range(num_evolve_layers)])
        self.fusion = CrossScholarFusion(mid_channels, out_channels)

        self.shortcut = None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True) if stride > 1 else nn.Identity(),
                CrossScholarFusion(in_channels, out_channels)
            )
        self.norm = LayerNorm2d(in_channels) if stride != 1 else None

    def forward(self, x):
        if self.norm is not None:
            x = self.norm(x)
        identity = self.shortcut(x) if self.shortcut is not None else x
        
        # 1. 膨胀下采样 (建立高维特征空间)
        x = self.expand(x)
        x = self.seblock(x)
        # 2. N层内部自演化 (通过内部残差保持记忆)
        x = self.evolution(x)
        # 3. 交叉注意力融合 (全局语义集成)
        x = self.fusion(x)
        return x + identity

