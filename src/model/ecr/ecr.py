import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from src.model.components.attention import SEBlock
from src.model.ecr.norm import RMSNorm2d, LayerNorm2d

class CrossScholarFusion(nn.Module):
    """
    交叉学者融合 (Cross Scholar Fusion)
    
    设计哲学:
    1. 语义投影 (W_K): 大小 16 x C，代表 16 种预定义的“语义主题”。
    2. 偏好检索 (W_Q): 大小 C x 16，代表通道对这 16 种主题的响应权重。
    3. 核心逻辑: 利用低秩近似 (Low-rank Approximation) 实现类似交叉注意力的全局通道融合。
    4. 计算优化: 利用结合律 W_Q @ (W_K @ X)，将计算复杂度从 O(C^2) 降低到 O(C * L)，其中 L 为潜空间维度 (16)。
    """
    def __init__(self, in_channels, out_channels, latent_dim=None):
        super().__init__()
        if latent_dim is None:
            latent_dim = in_channels // 16
        if latent_dim < 8:
            latent_dim = 8
        self.latent_dim = latent_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # W_K: 16 x C (分类/投影矩阵)
        self.W_K = nn.Parameter(torch.randn(latent_dim, in_channels) * 0.02)
        # W_Q: C x 16 (检索/还原矩阵)
        self.W_Q = nn.Parameter(torch.randn(out_channels, latent_dim) * 0.02)
        
        self.norm = RMSNorm2d(latent_dim)
        self.act = nn.SiLU() # 去掉 inplace，兼容 Gradient Checkpointing

    def forward(self, x):
        # 1. 投影到潜空间 (分类): (B, latent_dim, H, W) = W_K @ X
        # 注意：这里的 C 必须等于 self.in_channels，否则 W_K 形状不匹配
        x = F.conv2d(x, self.W_K.view(self.latent_dim, self.in_channels, 1, 1))
        
        # 2. 非线性激活与归一化 (语义过滤)
        x = self.act(self.norm(x))
        
        # 3. 从潜空间检索并还原: (B, out_channels, H, W) = W_Q @ 潜空间
        # 注意：这里输出的是 out_channels，而不是输入的 C
        x = F.conv2d(x, self.W_Q.view(self.out_channels, self.latent_dim, 1, 1))
        
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
    def __init__(self, in_channels, out_channels, num_evolve_layers=3, expansion=4, stride=1, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.stride = stride
        
        # mid_channels 为最终的输出通道数，由 expansion 决定
        mid_channels = in_channels * expansion
        
        # 0. Shortcut 路径 (保证梯度回传的黄金通道)
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True) if stride > 1 else nn.Identity(),
                CrossScholarFusion(in_channels, out_channels),
                LayerNorm2d(out_channels)
            )

        # 1. 分组膨胀与下采样 (Expansion & Downsample)
        # 用极低成本的分组卷积实现通道暴涨，物理隔离通道
        self.expand = nn.Sequential(
            LayerNorm2d(in_channels),
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False)
        )
        
        self.seblock = SEBlock(mid_channels, reduction=4)

        # 2. 高效演化 (Evolution)
        # 堆叠 N 层深度残差，积累空间特征记忆
        self.evolution = nn.Sequential(*[
            EfficientEvolutionLayer(mid_channels, kernel_size=3) for _ in range(num_evolve_layers)
        ])
        
        # 3. 交叉融合 (Thinking Fusion)
        # 使用低秩交叉注意力机制进行全局通道整合
        self.fusion = CrossScholarFusion(mid_channels, out_channels)

        # 4. 自动优化演化层 (根据系统环境选择最佳编译方案)
        # self._optimize_evolution()

    def forward(self, x):
        # 0. 黄金通道 (Shortcut)
        identity = self.shortcut(x)
        # 1. 膨胀下采样 (建立高维特征空间)
        x = self.expand(x)
        x = self.seblock(x)
        # 2. N层内部自演化 (通过内部残差保持记忆)
        if self.use_checkpoint and self.training:
            x = checkpoint(self.evolution, x, use_reentrant=False)
        else:
            x = self.evolution(x)
        # 3. 交叉注意力融合 (全局语义集成)
        x = self.fusion(x)
        # 4. 最终集成: 演化结果 + 原始映射
        return x + identity

    def _optimize_evolution(self):
        """尝试自动优化演化层 (解决 Windows 下 compile 不稳定问题)"""
        import platform
        is_windows = platform.system() == "Windows"
        try:
            if is_windows:
                 # Gradient Checkpointing 与 JIT/Compile 组合在旧版本中不稳定
                if self.use_checkpoint: return
                # Windows 下优先使用稳定的 torch.jit.script
                self.evolution = torch.jit.script(self.evolution)
            elif hasattr(torch, "compile"):
                # Linux/Unix 下优先使用强大的 torch.compile
                self.evolution = torch.compile(self.evolution)
            else:
                self.evolution = torch.jit.script(self.evolution)
        except Exception:
            pass # 如果优化失败，回退到原始模型

