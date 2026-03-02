import torch
import torch.nn.functional as F


# 定义损失函数：变分下界 (ELBO)
def loss_function(recon_x, x, mu, logvar, beta=1.0, loss_type='bce'):
    # 1. 重建损失：使用 sum 是为了满足 ELBO 对数似然定义，并与 KL 散度的量级平衡
    if loss_type == 'bce':
        recon_loss = F.binary_cross_entropy(recon_x, x.view(recon_x.size()), reduction='sum')
    else:
        recon_loss = F.mse_loss(recon_x, x.view(recon_x.size()), reduction='sum')

    # 2. KL 散度 (Kullback-Leibler Divergence)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # 总损失 = 重建损失 + beta * KL 散度
    return recon_loss + beta * KLD, recon_loss, KLD

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
