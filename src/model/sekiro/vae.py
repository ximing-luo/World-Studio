import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.components.resnet import BasicBlock

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
    """卷积版 Sekiro VAE (适用于 136x240)"""
    def __init__(self, latent_dim=256):
        super(ConvSekiroVAE, self).__init__(latent_dim)
        
        # Encoder: (3, 136, 240) -> (512, 4, 7)
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),   # 68x120
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # 34x60
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), # 17x30
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
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, output_padding=(1, 0)), # 17x30
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), # 34x60
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), # 68x120
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1), # 136x240
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
    def __init__(self, latent_dim=256, block=BasicBlock, num_blocks=[2, 2, 2, 2]):
        super(ResNetSekiroVAE, self).__init__(latent_dim)
        self.in_channels = 32
        self.block_expansion = block.expansion
        
        # Encoder
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1), # 68x120
            nn.GroupNorm(8, 32),
            nn.SiLU(inplace=True)
        )
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=2)  # 34x60
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2) # 17x30
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2) # 9x15
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2) # 5x8
        
        self.flatten_dim = 512 * self.block_expansion * 5 * 8
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)
        
        # Decoder
        self.fc_z = nn.Linear(latent_dim, self.flatten_dim)
        self.in_channels = 512 * self.block_expansion
        
        self.layer5 = self._make_layer(block, 512, num_blocks[3], stride=1)
        self.upsample1 = nn.ConvTranspose2d(512 * self.block_expansion, 256, kernel_size=3, stride=2, padding=1, output_padding=0) # 9x15
        
        self.in_channels = 256 * self.block_expansion
        self.layer6 = self._make_layer(block, 256, num_blocks[2], stride=1)
        self.upsample2 = nn.ConvTranspose2d(256 * self.block_expansion, 128, kernel_size=3, stride=2, padding=1, output_padding=(0, 1)) # 17x30
        
        self.in_channels = 128 * self.block_expansion
        self.layer7 = self._make_layer(block, 128, num_blocks[1], stride=1)
        self.upsample3 = nn.ConvTranspose2d(128 * self.block_expansion, 64, kernel_size=3, stride=2, padding=1, output_padding=(1, 1)) # 34x60
        
        self.in_channels = 64 * self.block_expansion
        self.layer8 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.upsample4 = nn.ConvTranspose2d(64 * self.block_expansion, 32, kernel_size=3, stride=2, padding=1, output_padding=(1, 1)) # 68x120
        
        self.upsample5 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=(1, 1)) # 136x240
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
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z):
        h = self.fc_z(z).view(-1, 512 * self.block_expansion, 5, 8)
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
