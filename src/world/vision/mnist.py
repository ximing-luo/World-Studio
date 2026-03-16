import torch
import torch.nn as nn
from src.model.components.resnet import BasicBlock, TraditionalBasicBlock, BottleNeck

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
            nn.Conv2d(in_channels, 32, 4, stride=2, padding=1), # 14x14
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), # 7x7
            nn.ReLU(),
        )
            
        # 解码器 (Decoder)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), # 14x14
            nn.ReLU(),
            nn.ConvTranspose2d(32, in_channels, 4, stride=2, padding=1), # 28x28
            nn.Sigmoid()
        )

    def encode(self, x):
        return self.encoder_conv(x)

    def decode(self, h):
        return self.decoder_conv(h)

class MNISTResNet(BaseVision):
    """残差 (ResNet) MNIST 视觉模块：仅负责特征提取与重构。"""
    def __init__(self, in_channels=1, block=BottleNeck, num_blocks=[2, 2], **block_kwargs):
        super().__init__()
        # 获取残差块的输出膨胀系数，默认为 1
        expansion = getattr(block, 'expansion', 1)
        self.in_channels = 64
        
        # 编码器 (Encoder)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.SiLU(inplace=True)
        )
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=2, **block_kwargs)  # 14x14
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, **block_kwargs) # 7x7
            
        # 解码器 (Decoder)
        self.in_channels = 128 * expansion
        self.layer3 = self._make_layer(block, 128, num_blocks[1], stride=1, **block_kwargs)
        self.upsample1 = nn.ConvTranspose2d(128 * expansion, 64 * expansion, kernel_size=3, stride=2, padding=1, output_padding=1) # 14x14
        
        self.in_channels = 64 * expansion
        self.layer4 = self._make_layer(block, 64, num_blocks[0], stride=1, **block_kwargs)
        self.upsample2 = nn.ConvTranspose2d(64 * expansion, 32, kernel_size=3, stride=2, padding=1, output_padding=1) # 28x28
        
        self.final_conv = nn.Sequential(
            nn.Conv2d(32, in_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def _make_layer(self, block, out_channels, num_block, stride, **kwargs):
        strides = [stride] + [1] * (num_block - 1)
        layers = []
        # 获取 expansion 类属性，默认为 1 (表示输出通道数与 out_channels 一致)
        expansion = getattr(block, 'expansion', 1)
        for s in strides:
            layers.append(block(self.in_channels, out_channels, stride=s, **kwargs))
            self.in_channels = out_channels * expansion
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
