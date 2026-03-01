import torch
from src.model.components.loss import loss_function


# 训练函数
def train(epoch, model, train_loader, optimizer, device, beta=1.0, loss_type='bce'):
    model.train()
    train_loss = 0
    recon_total_loss = 0
    kld_loss = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        
        # 计算损失
        loss, recon_loss, KLD = loss_function(recon_batch, target, mu, logvar, beta, loss_type=loss_type)
        loss.backward()
        
        train_loss += loss.item()
        recon_total_loss += recon_loss.item()
        kld_loss += KLD.item()
        optimizer.step()
        
        # 根据数据量调整打印频率
        log_interval = max(1, len(train_loader) // 10)
        if batch_idx % log_interval == 0:
            pixels_per_sample = data[0].numel()
            latent_dim = mu.size(1)
            recon_per_pixel = recon_loss.item() / (len(data) * pixels_per_sample)
            kld_per_dim = (KLD.item() / len(data)) / latent_dim
            
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
                  f'Loss: {loss.item() / len(data):.4f} ({loss_type.upper()}/px: {recon_per_pixel:.4f}, KLD/dim: {kld_per_dim:.4f})')
    
    avg_loss = train_loss / len(train_loader.dataset)
    avg_recon = recon_total_loss / len(train_loader.dataset)
    avg_kld = kld_loss / len(train_loader.dataset)
    
    sample_pixels = data[0].numel()
    latent_dim = mu.size(1)
    avg_recon_per_px = avg_recon / sample_pixels
    avg_kld_per_dim = avg_kld / latent_dim
    
    print(f'====> Epoch: {epoch} Average loss: {avg_loss:.4f} ({loss_type.upper()}/px: {avg_recon_per_px:.4f}, KLD/dim: {avg_kld_per_dim:.4f})')
    return avg_loss, avg_recon_per_px, avg_kld_per_dim