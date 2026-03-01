import torch
import torch.nn.functional as F


# 定义损失函数：变分下界 (ELBO)
def loss_function(recon_x, x, mu, logvar, beta=1.0, loss_type='bce'):
    # 1. 重建损失 (Reconstruction Loss)：衡量重建图与原图的差异
    if loss_type == 'bce':
        recon_loss = F.binary_cross_entropy(recon_x, x.view(recon_x.size()), reduction='sum')
    else:
        recon_loss = F.mse_loss(recon_x, x.view(recon_x.size()), reduction='sum')

    # 2. KL 散度 (Kullback-Leibler Divergence)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # 总损失 = 重建损失 + beta * KL 散度
    return recon_loss + beta * KLD, recon_loss, KLD
