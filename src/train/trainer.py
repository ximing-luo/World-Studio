import torch
from src.model.loss import loss_function


# 训练函数
def train(epoch, model, train_loader, optimizer, device, beta=1.0):
    model.train()
    train_loss = 0
    bce_loss = 0
    kld_loss = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        
        # 计算损失：重建目标变为旋转后的图
        loss, BCE, KLD = loss_function(recon_batch, target, mu, logvar, beta)
        loss.backward()
        
        train_loss += loss.item()
        bce_loss += BCE.item()
        kld_loss += KLD.item()
        optimizer.step()
        
        # 根据数据量调整打印频率：每 10 个 batch 或每 epoch 打印 4 次左右
        log_interval = max(1, len(train_loader) // 4)
        if batch_idx % log_interval == 0:
            # 计算归一化指标：每个像素的 BCE 和每个隐层维度的 KLD
            pixels_per_sample = data[0].numel()
            latent_dim = mu.size(1)
            bce_per_pixel = BCE.item() / (len(data) * pixels_per_sample)
            kld_per_dim = (KLD.item() / len(data)) / latent_dim
            
            # 打印进度：包含总损失、BCE/px、KLD/dim
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
                  f'Loss: {loss.item() / len(data):.4f} (BCE/px: {bce_per_pixel:.4f}, KLD/dim: {kld_per_dim:.4f})')
    
    avg_loss = train_loss / len(train_loader.dataset)
    avg_bce = bce_loss / len(train_loader.dataset)
    avg_kld = kld_loss / len(train_loader.dataset)
    
    # 获取归一化参数用于最终汇总
    sample_pixels = data[0].numel()
    latent_dim = mu.size(1)
    avg_bce_per_px = avg_bce / sample_pixels
    avg_kld_per_dim = avg_kld / latent_dim
    
    print(f'====> Epoch: {epoch} Average loss: {avg_loss:.4f} (BCE/px: {avg_bce_per_px:.4f}, KLD/dim: {avg_kld_per_dim:.4f})')
    return avg_loss, avg_bce_per_px, avg_kld_per_dim