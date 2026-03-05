import torch
import torch.nn as nn
from src.model.components.resnet import BottleNeck
from src.model.ecr.ecr import EfficientCrossResBlock
from .base import BaseVision, ThinkingSpace

class SekiroConv(BaseVision):
    """卷积 (CNN) Sekiro 视觉模块：仅负责特征提取与重构。"""
    def __init__(self, in_channels=3):
        super().__init__()
        
        # 编码器 (Encoder)
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, stride=2, padding=1),   # 64x120
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # 32x60
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), # 16x30
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1), # 8x15
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, stride=2, padding=1), # 4x7
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
            
        # 解码器 (Decoder)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, output_padding=(0, 1)), # 8x15
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), # 16x30
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), # 32x60
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), # 64x120
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(32, in_channels, 4, stride=2, padding=1), # 128x240
            nn.Sigmoid()
        )

    def encode(self, x):
        return self.encoder_conv(x)

    def decode(self, h):
        return self.decoder_conv(h)

class SekiroResNet(BaseVision):
    """残差 (ResNet) Sekiro 视觉模块：仅负责特征提取与重构。"""
    def __init__(self, in_channels=3, block=BottleNeck, num_blocks=[2, 2, 2, 2]):
        super().__init__()
        self.in_channels = 32
        
        # 编码器 (Encoder)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1), # 64x120
            nn.GroupNorm(8, 32),
            nn.SiLU(inplace=True)
        )
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=2)  # 32x60
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2) # 16x30
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2) # 8x15
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2) # 4x8
            
        # 解码器 (Decoder)
        self.in_channels = 512 * block.expansion
        self.layer5 = self._make_layer(block, 512, num_blocks[3], stride=1)
        self.upsample1 = nn.ConvTranspose2d(512 * block.expansion, 256 * block.expansion, kernel_size=3, stride=2, padding=1, output_padding=(1, 0)) # 8x15
        
        self.in_channels = 256 * block.expansion
        self.layer6 = self._make_layer(block, 256, num_blocks[2], stride=1)
        self.upsample2 = nn.ConvTranspose2d(256 * block.expansion, 128 * block.expansion, kernel_size=3, stride=2, padding=1, output_padding=(1, 1)) # 16x30
        
        self.in_channels = 128 * block.expansion
        self.layer7 = self._make_layer(block, 128, num_blocks[1], stride=1)
        self.upsample3 = nn.ConvTranspose2d(128 * block.expansion, 64 * block.expansion, kernel_size=3, stride=2, padding=1, output_padding=(1, 1)) # 32x60
        
        self.in_channels = 64 * block.expansion
        self.layer8 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.upsample4 = nn.ConvTranspose2d(64 * block.expansion, 32, kernel_size=3, stride=2, padding=1, output_padding=(1, 1)) # 64x120
        
        self.upsample5 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=(1, 1)) # 128x240
        self.final_conv = nn.Sequential(
            nn.Conv2d(16, in_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def _make_layer(self, block, out_channels, num_block, stride):
        strides = [stride] + [1] * (num_block - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, stride=s))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def encode(self, x):
        h = self.stem(x)
        h = self.layer1(h)
        h = self.layer2(h)
        h = self.layer3(h)
        h = self.layer4(h)
        return h

    def decode(self, h):
        h = self.layer5(h)
        h = self.upsample1(h)
        h = self.layer6(h)
        h = self.upsample2(h)
        h = self.layer7(h)
        h = self.upsample3(h)
        h = self.layer8(h)
        h = self.upsample4(h)
        h = self.upsample5(h)
        return self.final_conv(h)

class SekiroBrain(BaseVision):
    """大脑化残差版 Sekiro 视觉模块：集成编码和解码逻辑。"""
    def __init__(self, in_channels=3, latent_dim=256, block=EfficientCrossResBlock, num_blocks=[1, 1, 2, 1], is_vae=True):
        super().__init__()
        self.is_vae = is_vae
        self.in_channels = 32
        
        # 编码器 (Encoder)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1), # 64x120
            nn.GroupNorm(8, 32),
            nn.SiLU(inplace=True)
        )
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=2)  # 32x60
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2) # 16x30
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2) # 8x15
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2) # 4x8
        self.thinking_encoder = ThinkingSpace(channels=512, height=4, width=8)
        
        self.latent_channels = max(1, latent_dim // (4 * 8))
        if is_vae:
            # 空间潜空间: 兼容 VAE
            self.fc_mu = nn.Conv2d(512, self.latent_channels, 1)
            self.fc_logvar = nn.Conv2d(512, self.latent_channels, 1)
        else:
            # 空间潜空间: 兼容 VQ-VAE/JEPA/RSSM
            self.fc_latent = nn.Conv2d(512, self.latent_channels, 1)
            
        # 解码器 (Decoder)
        self.fc_z = nn.Conv2d(self.latent_channels, 512, 1)
        self.thinking_decoder = ThinkingSpace(channels=512, height=4, width=8, num_layers=8)
        self.in_channels = 512
        self.layer5 = self._make_layer(block, 512, num_blocks[3], stride=1)
        self.upsample1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=(1, 0)) # 8x15
        self.in_channels = 256
        self.layer6 = self._make_layer(block, 256, num_blocks[2], stride=1)
        self.upsample2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=(1, 1)) # 16x30
        self.in_channels = 128
        self.layer7 = self._make_layer(block, 128, num_blocks[1], stride=1)
        self.upsample3 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=(1, 1)) # 32x60
        self.in_channels = 64
        self.layer8 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.upsample4 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=(1, 1)) # 64x120
        self.upsample5 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=(1, 1)) # 128x240
        self.final_conv = nn.Sequential(
            nn.Conv2d(16, in_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def _make_layer(self, block, out_channels, num_block, stride):
        strides = [stride] + [1] * (num_block - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, stride=s))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def encode(self, x):
        h = self.stem(x)
        h = self.layer1(h)
        h = self.layer2(h)
        h = self.layer3(h)
        h = self.layer4(h)
        h = self.thinking_encoder(h)
        if self.is_vae:
            return self.fc_mu(h), self.fc_logvar(h)
        return self.fc_latent(h)

    def decode(self, z):
        # 支持一维或二维潜变量输入
        if len(z.shape) == 2:
            h = z.view(-1, self.latent_channels, 4, 8)
        else:
            h = z
        h = self.fc_z(h)
        h = self.thinking_decoder(h)
        h = self.layer5(h)
        h = self.upsample1(h)
        h = self.layer6(h)
        h = self.upsample2(h)
        h = self.layer7(h)
        h = self.upsample3(h)
        h = self.layer8(h)
        h = self.upsample4(h)
        h = self.upsample5(h)
        return self.final_conv(h)
