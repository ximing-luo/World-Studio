import torch
import torch.nn as nn
from typing import Dict, Tuple
from src.model.components import BasicBlock, BottleNeck, SEBlock
import torch.nn.functional as F

class Focus(nn.Module):
    """
    Focus layer (Space-to-Depth): 将空间信息切片并堆叠到通道维度。
    作用：无损下采样，大幅降低计算量。
    输入：(B, C, H, W) -> 输出：(B, C*4, H/2, W/2)
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return torch.cat([
            x[..., ::2, ::2],
            x[..., 1::2, ::2],
            x[..., ::2, 1::2],
            x[..., 1::2, 1::2]
        ], dim=1)

class SekiroEncoder(nn.Module):
    """
    VAE 编码器：基于 ResNet 和 Focus 架构。
    """
    def __init__(self, in_channels=3, latent_dim=512):
        super().__init__()
        self.in_channels = 32  # Stem 输出通道数
        
        # 1. 输入模块：Focus + 1x1 卷积
        # 预处理：(136, 240) -> Focus -> (68, 120) -> (32, 68, 120)
        self.stem = nn.Sequential(
            Focus(),
            nn.Conv2d(in_channels * 4, 32, kernel_size=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.SiLU(inplace=True)
        )
        
        # 2. 残差层阶段 (逐渐降低分辨率，增加通道)
        self.layer1 = self._make_layer(BasicBlock, 32, 1, 1)  # (32, 68, 120)
        self.layer2 = self._make_layer(BasicBlock, 64, 2, 2)  # (64, 34, 60)
        self.layer3 = self._make_layer(BasicBlock, 128, 4, 2) # (128, 17, 30)
        self.layer4 = self._make_layer(BasicBlock, 128, 2, 2) # (256, 9, 15)
        self.spatial_pool = nn.AdaptiveAvgPool2d((3, 5))
        
        # 3. 隐空间投影
        self.flatten_dim = 128 * 3 * 5 # 34560
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

    def _make_layer(self, block, out_channels, num_block, stride):
        strides = [stride] + [1] * (num_block - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, s))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        # 归一化：将 0-255 像素值缩放到 0-1
        if x.max() > 1.0:
            x = x.float() / 255.0
            
        h = self.stem(x)
        h = self.layer1(h)
        h = self.layer2(h)
        h = self.layer3(h)
        h = self.layer4(h)
        h = self.spatial_pool(h)
        h = h.flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)

class SekiroDecoder(nn.Module):
    """
    VAE 解码器：镜像编码器结构。
    """
    def __init__(self, latent_dim=512, out_channels=3):
        super().__init__()
        self.latent_dim = latent_dim
        self.initial_shape = (128, 3, 5)
        self.flatten_dim = 128 * 3 * 5
        
        self.fc = nn.Linear(latent_dim, self.flatten_dim)
        
        # 逆卷积/上采样阶段
        self.up5 = self._make_up_layer(128, 128, (9, 15))
        # Stage 4: (256, 9, 15) -> (128, 17, 30)
        self.up4 = self._make_up_layer(128, 128, (17, 30))
        # Stage 3: (128, 17, 30) -> (64, 34, 60)
        self.up3 = self._make_up_layer(128, 64, (34, 60))
        # Stage 2: (64, 34, 60) -> (32, 68, 120)
        self.up2 = self._make_up_layer(64, 32, (68, 120))
        # Stage 1: (32, 68, 120) -> (3, 136, 240)
        self.up1 = nn.Sequential(
            nn.Upsample(size=(136, 240), mode='bilinear', align_corners=False),
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def _make_up_layer(self, in_c, out_c, size):
        return nn.Sequential(
            nn.Upsample(size=size, mode='bilinear', align_corners=False),
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_c),
            nn.SiLU(inplace=True)
        )

    def forward(self, z) -> torch.Tensor:
        h = self.fc(z)
        h = h.view(-1, *self.initial_shape)
        h = self.up5(h)
        h = self.up4(h)
        h = self.up3(h)
        h = self.up2(h)
        return self.up1(h)

class SekiroVAE(nn.Module):
    """
    Sekiro 世界模型核心：基于稳定架构的 VAE。
    """
    def __init__(self, latent_dim=512):
        super().__init__()
        self.encoder = SekiroEncoder(latent_dim=latent_dim)
        self.decoder = SekiroDecoder(latent_dim=latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar

