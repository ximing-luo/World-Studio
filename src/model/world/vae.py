import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseVAE(nn.Module):
    """
    VAE 基类。
    vision: 提供 encode/decode 特征图的能力。
    projection: 投影层（需设置 stochastic=True）。
    """
    def __init__(self, vision, projection):
        super(BaseVAE, self).__init__()
        self.vision = vision
        self.projection = projection

    def encode(self, x):
        """返回分布参数 mu 和 logvar"""
        h = self.vision.encode(x)
        # 投影层内部已经根据 stochastic=True 输出 2x 维度
        z_params = self.projection.encode(h)
        mu, logvar = torch.chunk(z_params, 2, dim=-1)
        return mu, logvar

    def decode(self, z):
        """从潜变量重构数据"""
        # z: (B, Tokens, Latent_Dim)
        h = self.projection.decode(z)
        return self.vision.decode(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
