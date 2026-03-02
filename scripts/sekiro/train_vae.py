import os
import sys
import time
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

# 添加项目根目录到路径
path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(path)

from src.datasets.sekiro import Sekiro_VAE_Dataset
from src.model.sekiro.vae import ConvSekiroVAE, ResNetSekiroVAE
from src.model.components.loss import loss_function
from src.train.train_utils import get_log_dir

def train(epoch, model, train_loader, optimizer, device, beta=1.0, latent_dim=256, writer=None):
    model.train()
    train_loss = 0
    start_time = time.time()
    
    for batch_idx, (data, target, action) in enumerate(train_loader):
        # 将 uint8 数据搬运到 GPU 后再进行浮点转换和归一化 (极致性能)
        data = data.to(device).float() / 255.0
        target = target.to(device).float() / 255.0
        B, C, H, W = data.size()
        
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        
        # 任务：输入当前帧，重建/预测下一帧 (Target 为 next_obs)
        loss, recon_loss, KLD = loss_function(recon_batch, target, mu, logvar, beta, loss_type='mse')
        loss.backward()
        
        train_loss += loss.item()
        optimizer.step()
        
        if batch_idx % 10 == 0:
            # 计算用于评估的平均值（不改变损失纲量）
            # 1. Pixel MSE: 平均每个像素的误差 (recon_loss 是 sum 模式，除以 B*C*H*W)
            pixel_mse = recon_loss.item() / (B * C * H * W)
            # 2. KL per Dim: 平均每个隐维度的 KL (KLD 是 sum 模式，除以 B*latent_dim)
            kl_per_dim = KLD.item() / (B * latent_dim)
            
            elapsed = time.time() - start_time
            print(f'[{elapsed:.2f}s] Train Epoch: {epoch} [{batch_idx * B}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
                  f'Loss: {loss.item() / B:.4f} (Pixel MSE: {pixel_mse:.6f}, KL/Dim: {kl_per_dim:.6f})')
            
            if writer is not None:
                step = (epoch - 1) * len(train_loader) + batch_idx
                # Log per-sample loss
                writer.add_scalar('train/loss', loss.item() / len(data), step)
                writer.add_scalar('train/recon_loss', recon_loss.item() / len(data), step)
                writer.add_scalar('train/kld_loss', KLD.item() / len(data), step)
    
    avg_loss = train_loss / len(train_loader.dataset)
    print(f'====> Epoch: {epoch} Average loss: {avg_loss:.4f}')
    return avg_loss

def visualize(model, loader, device, epoch, writer=None):
    model.eval()
    with torch.no_grad():
        data, target, action = next(iter(loader))
        data = data.to(device).float() / 255.0
        target = target.to(device).float() / 255.0
        recon, _, _ = model(data)
        
        n = min(data.size(0), 5)
        fig, axes = plt.subplots(3, n, figsize=(n*3, 9))
        
        for i in range(n):
            # Input (Current Frame)
            img_in = data[i].cpu().permute(1, 2, 0).numpy()
            axes[0, i].imshow(img_in)
            axes[0, i].axis('off')
            if i == 0: axes[0, i].set_title('Input (t)')
            
            # Target (Future Frame)
            img_target = target[i].cpu().permute(1, 2, 0).numpy()
            axes[1, i].imshow(img_target)
            axes[1, i].axis('off')
            if i == 0: axes[1, i].set_title('Target (t+10)')
            
            # Reconstruction/Prediction
            img_recon = recon[i].cpu().permute(1, 2, 0).numpy()
            axes[2, i].imshow(img_recon)
            axes[2, i].axis('off')
            if i == 0: axes[2, i].set_title('Predicted (t+10)')
            
        plt.tight_layout()
        os.makedirs('outputs/results/sekiro/vae', exist_ok=True)
        plt.savefig(f'outputs/results/sekiro/vae/epoch_{epoch}.png')
        plt.close()
        print(f"Saved visualization to outputs/results/sekiro/vae/epoch_{epoch}.png")

def eval_task(model, loader, device, beta=1.0, writer=None, epoch=None):
    """
    验证任务：计算验证集上的重建损失和 KL 散度
    """
    model.eval()
    total_loss = 0
    total_recon_loss = 0
    total_kld = 0
    
    with torch.no_grad():
        for data, target, action in loader:
            data = data.to(device).float() / 255.0
            target = target.to(device).float() / 255.0
            
            recon_batch, mu, logvar = model(data)
            loss, recon_loss, KLD = loss_function(recon_batch, target, mu, logvar, beta, loss_type='mse')
            
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kld += KLD.item()
            
    # 注意 loss_function 返回的是 batch sum，除以 dataset size 得到 avg
    avg_loss = total_loss / len(loader.dataset)
    avg_recon = total_recon_loss / len(loader.dataset)
    avg_kld = total_kld / len(loader.dataset)
    
    print(f'====> Validation Loss: {avg_loss:.4f} (Recon: {avg_recon:.4f}, KL: {avg_kld:.4f})')
    
    if writer is not None and epoch is not None:
        writer.add_scalar('val/loss', avg_loss, epoch)
        writer.add_scalar('val/recon_loss', avg_recon, epoch)
        writer.add_scalar('val/kld_loss', avg_kld, epoch)
        
    return avg_loss

def main():
    # Hyperparameters
    batch_size = 32
    learning_rate = 1e-3
    epochs = 10
    latent_dim = 256
    beta = 1.0
    # Prediction offset (frame_skip in dataset)
    prediction_offset = 1 
    log_dir = get_log_dir('logs/sekiro/vae')

    results_dir = 'outputs/results/sekiro/vae'
    os.makedirs(results_dir, exist_ok=True)

    # DataLoader num_workers: Windows 下开启多进程会导致内存占用飙升
    num_workers = 2 
    
    # TensorBoard
    writer = SummaryWriter(log_dir)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | Task: Sekiro Next-Frame Prediction (VAE)")
    
    # 数据准备
    dataset = Sekiro_VAE_Dataset(data_dir='data/Sekiro/recordings', frame_skip=prediction_offset)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    # 模型初始化 - 支持 Conv, ResNet 架构
    # model = ResNetSekiroVAE(latent_dim=latent_dim).to(device)
    model = ConvSekiroVAE(latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(1, epochs + 1):
        train(epoch, model, train_loader, optimizer, device, beta, latent_dim, writer)
        eval_task(model, test_loader, device, beta, writer, epoch)
        visualize(model, train_loader, device, epoch, writer)

    # 保存模型
    os.makedirs('outputs/models', exist_ok=True)
    torch.save(model.state_dict(), 'outputs/models/sekiro_vae.pth')
    print(f"Model saved to 'outputs/models/sekiro_vae.pth'")
    
    writer.close()

if __name__ == '__main__':
    main()
