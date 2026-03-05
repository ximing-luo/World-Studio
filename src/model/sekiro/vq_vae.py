import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.mnist.vq_vae import VectorQuantizer
from src.model.components.resnet import BasicBlock

class BaseSekiroVQVAE(nn.Module):
    """Sekiro VQ-VAE 基类"""
    def __init__(self, num_embeddings=512, embedding_dim=64, commitment_cost=0.25):
        super(BaseSekiroVQVAE, self).__init__()
        self.vq_layer = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)

    def encode(self, x):
        raise NotImplementedError

    def decode(self, z):
        raise NotImplementedError

    def forward(self, x):
        z = self.encode(x)
        quantized, vq_loss, _ = self.vq_layer(z, is_spatial=len(z.shape) == 4)
        x_recon = self.decode(quantized)
        return x_recon, vq_loss

class ConvSekiroVQVAE(BaseSekiroVQVAE):
    """卷积版 Sekiro VQ-VAE (适用于 128x240)"""
    def __init__(self, in_channels=3, num_hiddens=128, num_embeddings=512, embedding_dim=64):
        super(ConvSekiroVQVAE, self).__init__(num_embeddings, embedding_dim)
        
        # Encoder: 128x240 -> 64x120 -> 32x60 -> 16x30
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(in_channels, num_hiddens//2, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, num_hiddens//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_hiddens//2, num_hiddens, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, num_hiddens),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_hiddens, embedding_dim, kernel_size=4, stride=2, padding=1),
            nn.Tanh() # 关键：将特征值限制在 (-1, 1) 之间，防止漂移过远导致 VQ Loss 爆炸
        )
        
        # Decoder: 16x30 -> 32x60 -> 64x120 -> 128x240
        self.decoder_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(embedding_dim, num_hiddens, kernel_size=3, padding=1),
            nn.GroupNorm(8, num_hiddens),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(num_hiddens, num_hiddens//2, kernel_size=3, padding=1),
            nn.GroupNorm(8, num_hiddens//2),
            nn.ReLU(inplace=True),
            # 最后一层：先通过卷积减少通道到 16，再上采样，这样显存占用能减小 ~80%
            nn.Conv2d(num_hiddens//2, 16, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(16, in_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        return self.encoder_conv(x)

    def decode(self, z):
        return self.decoder_conv(z)

class ResNetSekiroVQVAE(BaseSekiroVQVAE):
    """残差版 Sekiro VQ-VAE (适用于 128x240)"""
    def __init__(self, in_channels=3, num_hiddens=128, num_embeddings=512, embedding_dim=64, block=BasicBlock, num_blocks=[2, 2]):
        super(ResNetSekiroVQVAE, self).__init__(num_embeddings, embedding_dim)
        self.in_channels = num_hiddens
        self.block_expansion = block.expansion

        # Encoder: 128x240 -> 64x120 -> 32x60
        self.encoder_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, num_hiddens, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, num_hiddens),
            nn.SiLU(inplace=True)
        )
        self.layer1 = self._make_layer(block, num_hiddens, num_blocks[0], stride=2)  # 64x120
        self.layer2 = self._make_layer(block, embedding_dim // self.block_expansion, num_blocks[1], stride=2) # 32x60
        
        # Decoder: 32x60 -> 64x120 -> 128x240
        self.in_channels = embedding_dim
        self.layer3 = self._make_layer(block, embedding_dim // self.block_expansion, num_blocks[1], stride=1)
        self.upsample1 = nn.ConvTranspose2d(embedding_dim, num_hiddens, kernel_size=3, stride=2, padding=1, output_padding=1) # 64x120
        
        self.in_channels = num_hiddens
        self.layer4 = self._make_layer(block, num_hiddens, num_blocks[0], stride=1)
        self.upsample2 = nn.ConvTranspose2d(num_hiddens, num_hiddens // 2, kernel_size=3, stride=2, padding=1, output_padding=1) # 128x240

        self.final_conv = nn.Sequential(
            nn.Conv2d(num_hiddens // 2, in_channels, kernel_size=3, padding=1),
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
        h = self.encoder_conv1(x)
        h = self.layer1(h)
        return self.layer2(h)

    def decode(self, z):
        h = self.layer3(z)
        h = self.upsample1(h)
        h = self.layer4(h)
        h = self.upsample2(h)
        return self.final_conv(h)
