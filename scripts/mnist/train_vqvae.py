import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.model.mnist.vq_vae import FCVQVAE, ConvVQVAE, ResNetVQVAE
from src.datasets.mnist import MNIST_VQVAE_Dataset

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | Task: VQ-VAE Reconstruction")

    # 数据准备
    transform = transforms.ToTensor()
    # 使用标准 MNIST，VQ-VAE 主要任务是高质量重建/压缩
    mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_dataset = MNIST_VQVAE_Dataset(mnist_train)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)

    # 模型初始化 - 支持 FC, Conv, ResNet 三种架构
    # model = FCVQVAE(num_embeddings=512, embedding_dim=32).to(device)
    # model = ResNetVQVAE(in_channels=1, num_hiddens=64, num_embeddings=512, embedding_dim=32).to(device)
    model = ConvVQVAE(in_channels=1, num_hiddens=64, num_embeddings=512, embedding_dim=32).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    epochs = 10
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        total_recon = 0
        total_vq = 0
        
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            
            optimizer.zero_grad()
            
            # VQ-VAE forward 返回: reconstruction, vq_loss
            recon_batch, vq_loss = model(data)
            
            # 重建损失
            recon_loss = F.mse_loss(recon_batch, data)
            
            # 总损失
            loss = recon_loss + vq_loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_vq += vq_loss.item()
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch} [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f} (Recon: {recon_loss.item():.4f}, VQ: {vq_loss.item():.4f})")
        
        avg_loss = total_loss / len(train_loader)
        print(f"====> Epoch: {epoch} Average loss: {avg_loss:.4f}")
        
        # 可视化
        visualize_reconstruction(model, train_loader, device, epoch)

    # 保存模型
    os.makedirs('outputs/models', exist_ok=True)
    torch.save(model.state_dict(), 'outputs/models/mnist_vqvae.pth')
    print(f"Model saved to 'outputs/models/mnist_vqvae.pth'")

def visualize_reconstruction(model, loader, device, epoch, num_samples=8):
    model.eval()
    with torch.no_grad():
        data, _ = next(iter(loader))
        data = data[:num_samples].to(device)
        
        recon, _ = model(data)
        
        # 绘图: 上面是原图，下面是重建
        fig, axes = plt.subplots(2, num_samples, figsize=(num_samples*1.5, 3))
        
        for i in range(num_samples):
            # 原图
            ax = axes[0, i]
            ax.imshow(data[i].cpu().squeeze().numpy(), cmap='gray')
            ax.axis('off')
            if i == 0: ax.set_title("Original")
            
            # 重建
            ax = axes[1, i]
            ax.imshow(recon[i].cpu().squeeze().numpy(), cmap='gray')
            ax.axis('off')
            if i == 0: ax.set_title("Reconstruction")
            
        plt.tight_layout()
        os.makedirs('outputs/results/mnist/vqvae', exist_ok=True)
        plt.savefig(f'outputs/results/mnist/vqvae/epoch_{epoch}.png')
        plt.close()
        print(f"Saved visualization to outputs/results/mnist/vqvae/epoch_{epoch}.png")

if __name__ == "__main__":
    train()
