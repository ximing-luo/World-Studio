import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义适用于复杂图像（如只狼画面）的卷积 VAE 模型
class ConvVAE(nn.Module):
    def __init__(self, latent_dim=128):
        super(ConvVAE, self).__init__()
        self.latent_dim = latent_dim

        # 编码器：卷积层
        # 输入: (3, 136, 240)
        self.encoder_conv = nn.Sequential(
            # (3, 136, 240) -> (32, 68, 120)
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            
            # (32, 68, 120) -> (64, 34, 60)
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            # (64, 34, 60) -> (128, 17, 30)
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # (128, 17, 30) -> (256, 8, 15)
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # (256, 8, 15) -> (512, 4, 7)
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # 提取更抽象特征 (512, 4, 7) -> (128, 4, 7)
            nn.Conv2d(512, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # 展平维度: 128 * 4 * 7 = 3584
        dim = 128 * 4 * 7  
        self.fc_mu = nn.Linear(dim, latent_dim)
        self.fc_logvar = nn.Linear(dim, latent_dim)

        # 解码器：从隐空间重建
        self.decoder_fc = nn.Linear(latent_dim, dim)
        
        self.decoder_conv = nn.Sequential(
            # (128, 4, 7) -> (512, 4, 7)
            nn.Conv2d(128, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # (512, 4, 7) -> (256, 8, 15)
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, output_padding=(0, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # (256, 8, 15) -> (128, 17, 30)
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, output_padding=(1, 0), bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # (128, 17, 30) -> (64, 34, 60)
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # (64, 34, 60) -> (32, 68, 120)
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            # (32, 68, 120) -> (3, 136, 240)
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder_conv(x)
        h = h.reshape(h.size(0), -1) # 使用 reshape 替代 view
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_fc(z)
        h = h.reshape(h.size(0), 128, 4, 7) # 还原形状
        return self.decoder_conv(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
