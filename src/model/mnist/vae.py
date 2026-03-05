import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.components.resnet import BasicBlock, BottleNeck, EResBlock
from src.model.components.ecr import EfficientEvolutionLayer, EfficientCrossResBlock
from src.model.components.attention import ThinkingSpace

class BaseVAE(nn.Module):
    """VAE 基类，定义核心逻辑"""
    def __init__(self, latent_dim=20):
        super(BaseVAE, self).__init__()
        self.latent_dim = latent_dim

    def encode(self, x):
        raise NotImplementedError

    def decode(self, z):
        raise NotImplementedError

    def reparameterize(self, mu, logvar):
        """重参数化技巧：z = mu + eps * std"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class FCVAE(BaseVAE):
    """全连接 VAE (MLP)"""
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super(FCVAE, self).__init__(latent_dim)
        self.input_dim = input_dim
        self.encoder_fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder_fc(x.view(-1, self.input_dim))
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z):
        return self.decoder_fc(z).view(-1, 1, 28, 28)

class ConvVAE(BaseVAE):
    """卷积 VAE (CNN)"""
    def __init__(self, in_channels=1, latent_dim=20):
        super(ConvVAE, self).__init__(latent_dim)
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, stride=2, padding=1), # 14x14
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), # 7x7
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(64 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(64 * 7 * 7, latent_dim)
        self.fc_z = nn.Linear(latent_dim, 64 * 7 * 7)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), # 14x14
            nn.ReLU(),
            nn.ConvTranspose2d(32, in_channels, 4, stride=2, padding=1), # 28x28
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder_conv(x)
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z):
        h = self.fc_z(z).view(-1, 64, 7, 7)
        return self.decoder_conv(h)

class ResNetVAE(BaseVAE):
    """残差 VAE (ResNet 风格)"""
    def __init__(self, in_channels=1, block=BasicBlock, num_blocks=[2, 2], latent_dim=20):
        super(ResNetVAE, self).__init__(latent_dim)
        self.in_channels = 64
        self.block_expansion = block.expansion

        # Encoder
        self.encoder_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.SiLU(inplace=True)
        )
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=2)  # 14x14
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2) # 7x7
        
        flatten_dim = 128 * self.block_expansion * 7 * 7
        self.fc_mu = nn.Linear(flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(flatten_dim, latent_dim)

        # Decoder
        self.fc_z = nn.Linear(latent_dim, flatten_dim)
        self.in_channels = 128 * self.block_expansion
        self.layer3 = self._make_layer(block, 128, num_blocks[1], stride=1)
        self.upsample1 = nn.ConvTranspose2d(128 * self.block_expansion, 64 * self.block_expansion, kernel_size=2, stride=2)
        
        self.in_channels = 64 * self.block_expansion
        self.layer4 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.upsample2 = nn.ConvTranspose2d(64 * self.block_expansion, 32, kernel_size=2, stride=2)

        self.final_conv = nn.Sequential(
            nn.Conv2d(32, in_channels, kernel_size=3, padding=1),
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
        h = self.layer2(h)
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z):
        h = self.fc_z(z).view(-1, 128 * self.block_expansion, 7, 7)
        h = self.layer3(h)
        h = self.upsample1(h)
        h = self.layer4(h)
        h = self.upsample2(h)
        return self.final_conv(h)

class BrainResNetMNISTVAE(BaseVAE):
    """大脑化残差版 MNIST VAE (基于 EfficientCrossResBlock)"""
    def __init__(self, in_channels=1, block=EfficientCrossResBlock, num_blocks=[1, 1], latent_dim=20):
        super(BrainResNetMNISTVAE, self).__init__(latent_dim)
        self.in_channels = 32
        
        # Encoder
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1), # 28x28
            nn.GroupNorm(8, 32),
            nn.SiLU(inplace=True)
        )
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=2)  # 14x14
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2) # 7x7
        
        # 3. 思维空间: 在 7x7 的高维语义图上进行 DeepSeek-V3 思考
        self.thinking = ThinkingSpace(channels=128, height=7, width=7)
        
        self.flatten_dim = 128 * 7 * 7
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)
        
        # Decoder
        self.fc_z = nn.Linear(latent_dim, self.flatten_dim)
        self.in_channels = 128
        
        self.layer3 = self._make_layer(block, 128, num_blocks[1], stride=1)
        self.upsample1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=(1, 1)) # 14x14
        
        self.in_channels = 64
        self.layer4 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.upsample2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=(1, 1)) # 28x28
        
        self.final_conv = nn.Sequential(
            nn.Conv2d(32, in_channels, kernel_size=3, padding=1),
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
        
        # 深度思考: [B, 128, 7, 7] -> [B, 128, 7, 7]
        h = self.thinking(h)
        
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z):
        h = self.fc_z(z).view(-1, 128, 7, 7)
        h = self.layer3(h)
        h = self.upsample1(h)
        h = self.layer4(h)
        h = self.upsample2(h)
        return self.final_conv(h)


