import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class PerceptualLoss(nn.Module):
    """
    感知损失 (Perceptual Loss) - 基于超轻量 SqueezeNet 1.1 特征提取
    逻辑: 衡量生成图与原图在特征空间中的距离，相比 VGG16 极大节省显存。
    """
    def __init__(self, model_type='squeezenet'):
        super().__init__()
        # 使用预训练的 SqueezeNet 1.1
        model = models.squeezenet1_1(weights='DEFAULT').features
        
        # 截取不同阶段的特征提取层 (Fire Modules)
        # SqueezeNet 1.1 的结构: Conv -> Pool -> Fire2 -> Fire3 -> Pool -> Fire4 -> Fire5 -> Pool ...
        self.slice1 = nn.Sequential(*model[:2])   # Conv + ReLU
        self.slice2 = nn.Sequential(*model[2:5])  # MaxPool + Fire2 + Fire3
        self.slice3 = nn.Sequential(*model[5:8])  # MaxPool + Fire4 + Fire5
        
        # 冻结参数
        for param in self.parameters():
            param.requires_grad = False
            
    def forward(self, x, y):
        # 确保输入在合适的范围内 (通常预训练模型期望 ImageNet 归一化)
        # 这里假设输入已在 [0, 1] 之间
        h1_x = self.slice1(x)
        h2_x = self.slice2(h1_x)
        h3_x = self.slice3(h2_x)
        
        h1_y = self.slice1(y)
        h2_y = self.slice2(h1_y)
        h3_y = self.slice3(h2_y)
        
        loss = F.l1_loss(h1_x, h1_y) + F.l1_loss(h2_x, h2_y) + F.l1_loss(h3_x, h3_y)
        return loss

# 定义损失函数：变分下界 (ELBO)
def loss_function(recon_x, x, mu, logvar, beta=1.0, loss_type='mse', perceptual_loss=None, lambda_p=1.0):
    # 1. 重建损失
    if loss_type == 'bce':
        recon_loss = F.binary_cross_entropy(recon_x, x.view(recon_x.size()), reduction='sum')
    else:
        recon_loss = F.mse_loss(recon_x, x.view(recon_x.size()), reduction='sum')

    # 2. KL 散度 (Kullback-Leibler Divergence)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # 3. 感知损失 (可选)
    p_loss = 0.0
    if perceptual_loss is not None:
        p_loss = perceptual_loss(recon_x, x)
        
    # 总损失 = 重建损失 + beta * KL 散度 + lambda_p * 感知损失
    total_loss = recon_loss + beta * KLD + lambda_p * p_loss
    return total_loss, recon_loss, KLD, p_loss

def kl_divergence(mu1: torch.Tensor, logvar1: torch.Tensor, mu2: torch.Tensor, logvar2: torch.Tensor) -> torch.Tensor:
    """
    计算两个高斯分布之间的 KL 散度。
    常用于 VAE 或 RSSM 的 Prior/Posterior 匹配。
    """
    var1 = torch.exp(logvar1)
    var2 = torch.exp(logvar2)
    # KL(N(mu1, var1) || N(mu2, var2))
    kl = 0.5 * (logvar2 - logvar1 + (var1 + (mu1 - mu2)**2) / var2 - 1.0)
    return kl.sum(dim=-1).mean()
