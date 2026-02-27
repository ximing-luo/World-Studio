# 定义损失函数：变分下界 (ELBO)
def loss_function(recon_x, x, mu, logvar, beta=1.0):
    # 1. 重建损失 (Reconstruction Loss)：衡量重建图与原图的差异
    # 使用 binary_cross_entropy，reduction='sum' 表示对所有像素求和
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    
    # 2. KL 散度 (Kullback-Leibler Divergence)：衡量隐分布与标准正态分布的差异
    # KLD = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # 总损失 = 重建损失 + beta * KL散度
    return BCE + beta * KLD, BCE, KLD
