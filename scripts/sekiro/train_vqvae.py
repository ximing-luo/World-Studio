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

from src.datasets.sekiro import Sekiro_VQVAE_Dataset
from src.model.sekiro.vq_vae import ConvSekiroVQVAE, ResNetSekiroVQVAE
from src.model.components.loss import PerceptualLoss
from src.model.components.discriminator import PatchGANDiscriminator, GANLoss
from src.train.train_utils import get_log_dir

def train(epoch, model, discriminator, train_loader, optimizer_G, optimizer_D, perceptual_loss_fn, gan_loss_fn, device, embedding_dim=64, writer=None):
    model.train()
    discriminator.train()
    total_loss_G = 0
    total_loss_D = 0
    start_time = time.time()
    
    # 权重系数
    lambda_p = 0.05   # 感知损失权重 (大幅降低，防止梯度爆炸)
    lambda_gan = 0.05 # 对抗损失权重
    
    for batch_idx, (data, target, action) in enumerate(train_loader):
        data = data.to(device).float() / 255.0
        target = target.to(device).float() / 255.0
        B, C, H, W = data.size()
        
        # ----------------------------
        # 1. 优化 VQ-VAE (生成器)
        # ----------------------------
        optimizer_G.zero_grad()
        
        # 重构/预测
        recon_batch, vq_loss = model(data)
        
        # (a) 基础重建损失
        recon_loss = F.mse_loss(recon_batch, target)
        
        # (b) 感知损失
        p_loss = perceptual_loss_fn(recon_batch, target)
        
        # (c) 对抗损失 (生成器部分)
        pred_fake = discriminator(recon_batch)
        g_loss = gan_loss_fn(pred_fake, True) # 希望判别器认为是真图
        
        # 汇总 G 损失
        loss_G = recon_loss + vq_loss + lambda_gan * g_loss + lambda_p * p_loss
        
        loss_G.backward()
        norm_G = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer_G.step()
        
        # ----------------------------
        # 2. 优化判别器
        # ----------------------------
        optimizer_D.zero_grad()
        
        # 判别真图
        pred_real = discriminator(target)
        loss_D_real = gan_loss_fn(pred_real, True)
        
        # 判别假图 (使用 detach 避免梯度回传到 VQ-VAE)
        pred_fake_det = discriminator(recon_batch.detach())
        loss_D_fake = gan_loss_fn(pred_fake_det, False)
        
        # 汇总 D 损失
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        
        loss_D.backward()
        norm_D = torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
        optimizer_D.step()
        
        total_loss_G += loss_G.item()
        total_loss_D += loss_D.item()
        
        if batch_idx % 10 == 0:
            elapsed = time.time() - start_time
            print(f'[{elapsed:.2f}s] Epoch: {epoch} [{batch_idx * B}/{len(train_loader.dataset)}]\t'
                  f'L_G: {loss_G.item():.4f} (Rec: {recon_loss.item():.4f}, VQ: {vq_loss.item():.4f}, P: {p_loss.item():.4f}, GAN: {g_loss.item():.4f})\t'
                  f'L_D: {loss_D.item():.4f} (Norm_D: {norm_D:.2f})')
            
            if writer is not None:
                step = (epoch - 1) * len(train_loader) + batch_idx
                writer.add_scalar('train/loss_G', loss_G.item(), step)
                writer.add_scalar('train/loss_D', loss_D.item(), step)
                writer.add_scalar('train/recon_loss', recon_loss.item(), step)
                writer.add_scalar('train/perceptual_loss', p_loss.item(), step)
                writer.add_scalar('train/gan_loss_G', g_loss.item(), step)
    
    avg_loss_G = total_loss_G / len(train_loader)
    print(f'====> Epoch: {epoch} Average G Loss: {avg_loss_G:.4f}')
    return avg_loss_G

def visualize(model, loader, device, epoch, writer=None):
    model.eval()
    with torch.no_grad():
        data, target, action = next(iter(loader))
        data = data.to(device).float() / 255.0
        target = target.to(device).float() / 255.0
        recon, _ = model(data)
        
        n = min(data.size(0), 5)
        fig, axes = plt.subplots(3, n, figsize=(n*3, 9))
        
        for i in range(n):
            # Input
            axes[0, i].imshow(data[i].cpu().permute(1, 2, 0).numpy())
            axes[0, i].axis('off')
            if i == 0: axes[0, i].set_title('Original (t)')
            
            # Target
            axes[1, i].imshow(target[i].cpu().permute(1, 2, 0).numpy())
            axes[1, i].axis('off')
            if i == 0: axes[1, i].set_title('Target (t+10)')
            
            # Reconstruction
            axes[2, i].imshow(recon[i].cpu().permute(1, 2, 0).numpy())
            axes[2, i].axis('off')
            if i == 0: axes[2, i].set_title('VQ-Predicted (t+10)')
            
        plt.tight_layout()
        os.makedirs('outputs/results/sekiro/vqvae', exist_ok=True)
        plt.savefig(f'outputs/results/sekiro/vqvae/epoch_{epoch}.png')
        plt.close()
        print(f"Saved visualization to outputs/results/sekiro/vqvae/epoch_{epoch}.png")

def eval_task(model, loader, device, writer=None, epoch=None, perceptual_loss_fn=None):
    """
    验证任务：计算验证集上的重建损失、VQ 损失和感知损失
    """
    model.eval()
    total_recon_loss = 0
    total_vq_loss = 0
    total_p_loss = 0
    total_loss = 0
    
    with torch.no_grad():
        for data, target, action in loader:
            data = data.to(device).float() / 255.0
            target = target.to(device).float() / 255.0
            
            recon_batch, vq_loss = model(data)
            recon_loss = F.mse_loss(recon_batch, target)
            
            p_loss = 0
            if perceptual_loss_fn is not None:
                p_loss = perceptual_loss_fn(recon_batch, target).item()
                total_p_loss += p_loss
            
            loss = recon_loss + vq_loss
            
            total_recon_loss += recon_loss.item()
            total_vq_loss += vq_loss.item()
            total_loss += loss.item()
            
    avg_loss = total_loss / len(loader)
    avg_recon = total_recon_loss / len(loader)
    avg_vq = total_vq_loss / len(loader)
    avg_p = total_p_loss / len(loader)
    
    print(f'====> Validation Loss: {avg_loss:.4f} (Recon: {avg_recon:.4f}, VQ: {avg_vq:.4f}, Perceptual: {avg_p:.4f})')
    
    if writer is not None and epoch is not None:
        writer.add_scalar('val/loss', avg_loss, epoch)
        writer.add_scalar('val/recon_loss', avg_recon, epoch)
        writer.add_scalar('val/vq_loss', avg_vq, epoch)
        writer.add_scalar('val/perceptual_loss', avg_p, epoch)
        
    return avg_loss

def main():
    # Hyperparameters
    batch_size = 32
    learning_rate = 1e-3
    epochs = 10
    prediction_offset = 0 # 设为 0，关闭预测偏移，变为自编码重建任务
    num_hiddens = 192
    num_embeddings = 1024  # 扩充单词表，提升细节表达能力
    embedding_dim = 64    # 特征维度保持 64，足以表达单个单词含义
    log_dir = get_log_dir('logs/sekiro/vqvae')
    
    # DataLoader num_workers: Windows 下开启多进程会导致内存占用飙升 (每个 worker 复制一份库文件内存)
    # 如果内存不足 (看到 7GB+)，建议设为 0 或 1；若内存充足想要加速，可设为 2-4
    num_workers = 0 

    results_dir = 'outputs/results/sekiro/vqvae'
    os.makedirs(results_dir, exist_ok=True)
    
    # TensorBoard Writer
    writer = SummaryWriter(log_dir)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | Task: Sekiro VQ-VAE Next-Frame Prediction")
    
    # 数据准备
    dataset = Sekiro_VQVAE_Dataset(frame_skip=prediction_offset, data_dir='data/Sekiro/recordings')
    # 简单划分训练集和验证集 (使用相同的数据集对象，但在 loader 中 shuffle 不同)
    # 在实际项目中应严格划分，这里为了演示方便复用 dataset
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    # 模型初始化 - 支持 Conv, ResNet 架构
    # model = ResNetSekiroVQVAE(in_channels=3, num_hiddens=num_hiddens, num_embeddings=num_embeddings, embedding_dim=embedding_dim).to(device)
    model = ConvSekiroVQVAE(in_channels=3, num_hiddens=num_hiddens, num_embeddings=num_embeddings, embedding_dim=embedding_dim).to(device)
    
    # 判别器与损失函数初始化
    discriminator = PatchGANDiscriminator(input_nc=3).to(device)
    perceptual_loss_fn = PerceptualLoss().to(device) # 切换到超轻量 SqueezeNet
    gan_loss_fn = GANLoss().to(device)
    
    optimizer_G = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999))
    
    for epoch in range(1, epochs + 1):
        train(epoch, model, discriminator, train_loader, optimizer_G, optimizer_D, 
              perceptual_loss_fn, gan_loss_fn, device, embedding_dim, writer)
        eval_task(model, test_loader, device, writer, epoch, perceptual_loss_fn)
        visualize(model, train_loader, device, epoch, writer)

    # 保存模型
    os.makedirs('outputs/models', exist_ok=True)
    torch.save(model.state_dict(), 'outputs/models/sekiro_vqvae.pth')
    print(f"Model saved to 'outputs/models/sekiro_vqvae.pth'")
    
    writer.close()

if __name__ == '__main__':
    main()
