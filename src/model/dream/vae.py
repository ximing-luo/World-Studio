import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.components.resnet import BasicBlock, BottleNeck

# 定义 VAE 模型
class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super(VAE, self).__init__()
        self.input_dim = input_dim

        # 编码器：将输入压缩到隐空间
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)  # 均值 mu
        self.fc22 = nn.Linear(hidden_dim, latent_dim)  # 对数方差 logvar

        # 解码器：从隐空间重建输入
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        # 重参数化技巧：z = mu + eps * std
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        # 自动展平输入
        mu, logvar = self.encode(x.view(-1, self.input_dim))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

class ResVAE(nn.Module):
    """
    Residual VAE (ResNet 风格) 用于手写数字生成 (28x28)
    支持更换基础块 (BasicBlock) 或瓶颈块 (BottleNeck)
    """
    def __init__(self, in_channels, block, num_blocks, latent_dim):
        super(ResVAE, self).__init__()

        self.in_channels = 64
        self.block_expansion = block.expansion

        # --- Encoder ---
        # 输入对齐模块: 28x28
        self.encoder_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.SiLU(inplace=True)
        )

        # 下采样两次: 28 -> 14 -> 7
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=2)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)

        # 隐空间投影
        flatten_dim = 128 * self.block_expansion * 7 * 7
        self.fc_mu = nn.Linear(flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(flatten_dim, latent_dim)

        # --- Decoder ---
        self.fc_z = nn.Linear(latent_dim, flatten_dim)

        # 逆过程
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

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_z(z)
        h = h.view(h.size(0), 128 * self.block_expansion, 7, 7)
        h = self.layer3(h)
        h = self.upsample1(h)
        h = self.layer4(h)
        h = self.upsample2(h)
        return self.final_conv(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def res20(block=BasicBlock, num_blocks=[6,4], latent_dim=20):
    return ResVAE(in_channels=1, block=block, num_blocks=num_blocks, latent_dim=latent_dim)

