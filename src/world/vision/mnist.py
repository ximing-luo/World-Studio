import torch
import torch.nn as nn
from src.model.components.resnet import BasicBlock, TraditionalBasicBlock, BottleNeck
from src.model.ecr import EcrBlock

from .base import BaseVision

class MNISTFC(BaseVision):
    """全连接 (MLP) MNIST 视觉模块：仅负责特征提取与重构。"""
    def __init__(self, input_dim=784, hidden_dim=400):
        super().__init__()
        self.input_dim = input_dim
        
        # 编码器 (Encoder)
        self.encoder_fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
            
        # 解码器 (Decoder)
        self.decoder_fc = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def encode(self, x):
        return self.encoder_fc(x.view(-1, self.input_dim))

    def decode(self, h):
        # h: (B, C, H, W) or (B, D)
        if len(h.shape) == 4:
            h = h.flatten(1)
        return self.decoder_fc(h).view(-1, 1, 28, 28)

class MNISTConv(BaseVision):
    """卷积 (CNN) MNIST 视觉模块：仅负责特征提取与重构。"""
    def __init__(self, in_channels=1):
        super().__init__()
        
        # 编码器 (Encoder)
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, stride=2, padding=1), # 14x14
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), # 7x7
            nn.ReLU(),
        )
            
        # 解码器 (Decoder)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), # 14x14
            nn.ReLU(),
            nn.ConvTranspose2d(64, in_channels, 4, stride=2, padding=1), # 28x28
            nn.Sigmoid()
        )

    def encode(self, x):
        return self.encoder_conv(x)

    def decode(self, h):
        return self.decoder_conv(h)

class MNISTResNet(BaseVision):
    """残差 (ResNet) MNIST 视觉模块：仅负责特征提取与重构。"""
    def __init__(self, in_channels=1, block=EcrBlock, num_blocks=[6, 6]):
        super().__init__()
        self.in_channels = 64
        
        # 编码器 (Encoder)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.SiLU(inplace=True)
        )
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=2)  # 14x14
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2) # 7x7
            
        # 解码器 (Decoder)
        self.layer3 = self._make_layer(block, 128, num_blocks[1], stride=1)
        self.upsample1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1) # 14x14
        self.in_channels = 64
        
        self.layer4 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.upsample2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1) # 28x28
        
        self.final_conv = nn.Sequential(
            nn.Conv2d(32, in_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def _make_layer(self, block, dim, num_blocks, stride=1):
        layers = []
        # 只有第一个块可能负责处理 stride (和可能的维度切换)
        layers.append(block(self.in_channels, dim, stride=stride))
        self.in_channels = dim # 这里的 dim 是该 Stage 预设的主干宽度
        
        # 后续所有块全是 dim -> dim
        for _ in range(1, num_blocks):
            layers.append(block(dim, dim, stride=1))
        return nn.Sequential(*layers)

    def encode(self, x):
        h = self.stem(x)
        h = self.layer1(h)
        h = self.layer2(h)
        return h

    def decode(self, h):
        h = self.layer3(h)
        h = self.upsample1(h)
        h = self.layer4(h)
        h = self.upsample2(h)
        return self.final_conv(h)
