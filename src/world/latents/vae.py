import torch
import torch.nn as nn
import torch.nn.functional as F

class VAELatent(nn.Module):
    """
    VAE 隐空间约束层。
    负责分布参数的拆分、重参数化采样以及 KL 散度计算。
    """
    def __init__(self):
        super().__init__()

    def forward(self, tokens):
        """
        tokens: (B, S, C*2) - 投影层输出的均值和方差参数。
        返回: 采样后的 z, 以及 KL 损失。
        """
        # 拆分 mu 和 logvar
        mu, logvar = torch.chunk(tokens, 2, dim=-1)
        
        # 重参数化采样
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        # 计算 KL 散度损失: 0.5 * sum(exp(logvar) + mu^2 - 1 - logvar)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        kl_loss = kl_loss.mean()
        
        return z, kl_loss

    def sample(self, mu, logvar):
        """仅执行采样"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
