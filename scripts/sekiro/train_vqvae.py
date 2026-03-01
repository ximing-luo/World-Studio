import os
import sys
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

# 添加项目根目录到路径
path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(path)

from src.datasets.sekiro import Sekiro_VQVAE_Dataset
from src.model.sekiro.vq_vae import ConvSekiroVQVAE, ResNetSekiroVQVAE

def train(epoch, model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    
    for batch_idx, (data, target, action) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        
        # 任务：输入当前帧，预测/重构下一帧
        recon_batch, vq_loss = model(data)
        
        # 重建损失 (对比下一帧)
        recon_loss = F.mse_loss(recon_batch, target)
        loss = recon_loss + vq_loss
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
                  f'Loss: {loss.item():.4f} (Recon: {recon_loss.item():.4f}, VQ: {vq_loss.item():.4f})')
    
    avg_loss = total_loss / len(train_loader)
    print(f'====> Epoch: {epoch} Average loss: {avg_loss:.4f}')
    return avg_loss

def visualize_reconstruction(model, loader, device, epoch):
    model.eval()
    with torch.no_grad():
        data, target, action = next(iter(loader))
        data, target = data.to(device), target.to(device)
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

def main():
    results_dir = 'outputs/results/sekiro/vqvae'
    os.makedirs(results_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | Task: Sekiro VQ-VAE Next-Frame Prediction")
    
    # 数据准备
    dataset = Sekiro_VQVAE_Dataset(frame_skip=1, data_dir='data/demos')
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    
    # 模型初始化 - 支持 Conv, ResNet 架构
    # model = ResNetSekiroVQVAE(in_channels=3, num_hiddens=128, num_embeddings=512, embedding_dim=64).to(device)
    model = ConvSekiroVQVAE(in_channels=3, num_hiddens=128, num_embeddings=512, embedding_dim=64).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(1, 11):
        train(epoch, model, train_loader, optimizer, device)
        visualize_reconstruction(model, train_loader, device, epoch)

    # 保存模型
    os.makedirs('outputs/models', exist_ok=True)
    torch.save(model.state_dict(), 'outputs/models/sekiro_vqvae.pth')
    print(f"Model saved to 'outputs/models/sekiro_vqvae.pth'")

if __name__ == '__main__':
    main()
