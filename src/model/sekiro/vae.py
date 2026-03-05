import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.components.resnet import BasicBlock, BottleNeck, ResBlock, EResBlock
from src.model.components.ecr import EfficientCrossResBlock
from src.model.components.attention import ThinkingSpace, AttentionPooling


class BaseSekiroVAE(nn.Module):
    """Sekiro VAE 基类"""
    def __init__(self, latent_dim=256):
        super(BaseSekiroVAE, self).__init__()
        self.latent_dim = latent_dim

    def encode(self, x):
        raise NotImplementedError

    def decode(self, z):
        raise NotImplementedError

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

class ConvSekiroVAE(BaseSekiroVAE):
    """卷积版 Sekiro VAE (适用于 128x240)"""
    def __init__(self, latent_dim=256):
        super(ConvSekiroVAE, self).__init__(latent_dim)
        
        # Encoder: (3, 128, 240) -> (512, 4, 7)
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),   # 64x120
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
            nn.Conv2d(512, 128, 3, stride=1, padding=1), # 4x7 (抽象特征)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.flatten_dim = 128 * 4 * 7
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)
        
        # Decoder
        self.decoder_fc = nn.Linear(latent_dim, self.flatten_dim)
        self.decoder_conv = nn.Sequential(
            nn.Conv2d(128, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
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
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1), # 128x240
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder_conv(x)
        h = h.flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z):
        h = self.decoder_fc(z).view(-1, 128, 4, 7)
        return self.decoder_conv(h)

class ResNetSekiroVAE(BaseSekiroVAE):
    """残差版 Sekiro VAE (基于 ResNet 块)"""
    def __init__(self, latent_dim=256, block=BottleNeck, num_blocks=[2, 2, 2, 2]):
        super(ResNetSekiroVAE, self).__init__(latent_dim)
        self.in_channels = 32
        self.block_expansion = block.expansion
        
        # Encoder
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1), # 64x120
            nn.GroupNorm(8, 32),
            nn.SiLU(inplace=True)
        )
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=2)  # 32x60
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2) # 16x30
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2) # 8x15
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2) # 4x8
        
        self.flatten_dim = 512 * self.block_expansion * 4 * 8
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)
        
        # Decoder
        self.fc_z = nn.Linear(latent_dim, self.flatten_dim)
        self.in_channels = 512 * self.block_expansion
        
        self.layer5 = self._make_layer(block, 512, num_blocks[3], stride=1)
        self.upsample1 = nn.ConvTranspose2d(512 * self.block_expansion, 256 * self.block_expansion, kernel_size=3, stride=2, padding=1, output_padding=(1, 0)) # 8x15
        
        self.in_channels = 256 * self.block_expansion
        self.layer6 = self._make_layer(block, 256, num_blocks[2], stride=1)
        self.upsample2 = nn.ConvTranspose2d(256 * self.block_expansion, 128 * self.block_expansion, kernel_size=3, stride=2, padding=1, output_padding=(1, 1)) # 16x30
        
        self.in_channels = 128 * self.block_expansion
        self.layer7 = self._make_layer(block, 128, num_blocks[1], stride=1)
        self.upsample3 = nn.ConvTranspose2d(128 * self.block_expansion, 64 * self.block_expansion, kernel_size=3, stride=2, padding=1, output_padding=(1, 1)) # 32x60
        
        self.in_channels = 64 * self.block_expansion
        self.layer8 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.upsample4 = nn.ConvTranspose2d(64 * self.block_expansion, 32, kernel_size=3, stride=2, padding=1, output_padding=(1, 1)) # 64x120
        
        self.upsample5 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=(1, 1)) # 128x240
        self.final_conv = nn.Sequential(
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
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
        
        # 深度思考: [B, 512, 4, 8] -> [B, 512, 4, 8]
        h = self.thinking_encoder(h)
        
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z):
        h = self.fc_z(z).view(-1, 512 * self.block_expansion, 4, 8)
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

class BrainResNetSekiroVAE(BaseSekiroVAE):
    """大脑化残差版 Sekiro VAE (基于 EfficientCrossResBlock)"""
    def __init__(self, latent_dim=256, block=EfficientCrossResBlock, num_blocks=[1, 1, 2, 1]):
        super(BrainResNetSekiroVAE, self).__init__(latent_dim)
        self.in_channels = 32
        
        # Encoder
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1), # 64x120
            nn.GroupNorm(8, 32),
            nn.SiLU(inplace=True)
        )
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=2)  # 32x60
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2) # 16x30
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2) # 8x15
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2) # 4x8
        
        # 3. 思维空间: 在 4x8 的高维语义图上进行 DeepSeek-V3 思考 (Encoder)
        self.thinking_encoder = ThinkingSpace(channels=512, height=4, width=8)
        
        # 方案 B (已注释): 注意力池化 - 提取全局抽象语义
        # self.attention_pooling = AttentionPooling(dim=512)
        # self.fc_mu = nn.Linear(512, latent_dim)
        # self.fc_logvar = nn.Linear(512, latent_dim)
        
        # 方案 C: 空间潜空间 (Spatial Latent) - 保持 4x8 的二维物理结构
        # 每一格代表原图一个区域的概率分布，对 RSSM 的空间推理极度友好
        self.latent_channels = max(1, latent_dim // (4 * 8)) 
        self.fc_mu = nn.Conv2d(512, self.latent_channels, 1)
        self.fc_logvar = nn.Conv2d(512, self.latent_channels, 1)
        
        # Decoder
        self.fc_z = nn.Conv2d(self.latent_channels, 512, 1)
        
        # 3. 思维空间: 在 4x8 的高维语义图上进行 DeepSeek-V3 思考 (Decoder)
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
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def _make_layer(self, block, out_channels, num_block, stride):
        strides = [stride] + [1] * (num_block - 1)
        layers = []
        for s in strides:
            # ResBlock 接口: (in_channels, out_channels, mid_channels=None, stride=1)
            # 我们不需要显式传递 mid_channels，让它内部默认 *2 或 *4 即可
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
        
        # 空间潜空间: 直接返回二维概率分布 [B, channels, 4, 8]
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z):
        # z shape: [B, latent_channels, 4, 8]
        h = self.fc_z(z)
        
        # 深度思考: [B, 512, 4, 8] -> [B, 512, 4, 8]
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
