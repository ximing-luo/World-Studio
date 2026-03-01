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

from src.datasets.sekiro import Sekiro_VAE_Dataset
from src.model.sekiro.vae import ConvSekiroVAE, ResNetSekiroVAE
from src.model.components.loss import loss_function

def train(epoch, model, train_loader, optimizer, device, beta=1.0):
    model.train()
    train_loss = 0
    
    for batch_idx, (data, target, action) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        
        # 任务：输入当前帧，重建/预测下一帧 (Target 为 next_obs)
        loss, recon_loss, KLD = loss_function(recon_batch, target, mu, logvar, beta, loss_type='mse')
        loss.backward()
        
        train_loss += loss.item()
        optimizer.step()
        
        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
                  f'Loss: {loss.item() / len(data):.4f}')
    
    avg_loss = train_loss / len(train_loader.dataset)
    print(f'====> Epoch: {epoch} Average loss: {avg_loss:.4f}')
    return avg_loss

def visualize_reconstruction(model, loader, device, epoch):
    model.eval()
    with torch.no_grad():
        data, target, action = next(iter(loader))
        data, target = data.to(device), target.to(device)
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

def main():
    results_dir = 'outputs/results/sekiro/vae'
    os.makedirs(results_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | Task: Sekiro Next-Frame Prediction (VAE)")
    
    # 数据准备
    dataset = Sekiro_VAE_Dataset(data_dir='data/demos')
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    
    # 初始化模型 - 支持 Conv, ResNet 架构
    # model = ResNetSekiroVAE(latent_dim=256).to(device)
    model = ConvSekiroVAE(latent_dim=256).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(1, 11):
        train(epoch, model, train_loader, optimizer, device)
        visualize_reconstruction(model, train_loader, device, epoch)

    # 保存模型
    os.makedirs('outputs/models', exist_ok=True)
    torch.save(model.state_dict(), 'outputs/models/sekiro_vae.pth')
    print(f"Model saved to 'outputs/models/sekiro_vae.pth'")

if __name__ == '__main__':
    main()
