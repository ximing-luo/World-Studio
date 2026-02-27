import os
import sys
import shutil
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# 添加项目根目录到路径
path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(path)

from src.datasets.dataset import RotatedMNIST
from src.model.dream.vae import res20, VAE
from src.model.loss import loss_function

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
        
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
                  f'Loss: {loss.item() / len(data):.4f}')
    
    avg_loss = train_loss / len(train_loader.dataset)
    print(f'====> Epoch: {epoch} Average loss: {avg_loss:.4f}')
    return avg_loss

def visualize_reconstruction(model, loader, device, epoch):
    model.eval()
    with torch.no_grad():
        data, target = next(iter(loader))
        data = data.to(device)
        target = target.to(device)
        recon, _, _ = model(data)
        
        # Take first 8 samples
        n = 8
        data = data[:n].cpu()
        target = target[:n].cpu()
        recon = recon[:n].cpu()
        
        fig, axes = plt.subplots(3, n, figsize=(n*1.5, 4.5))
        for i in range(n):
            # Input
            axes[0, i].imshow(data[i].squeeze(), cmap='gray')
            axes[0, i].axis('off')
            if i == 0: axes[0, i].set_title('Input')
            
            # Target (Rotated)
            axes[1, i].imshow(target[i].squeeze(), cmap='gray')
            axes[1, i].axis('off')
            if i == 0: axes[1, i].set_title('Target')
            
            # Reconstruction
            axes[2, i].imshow(recon[i].squeeze(), cmap='gray')
            axes[2, i].axis('off')
            if i == 0: axes[2, i].set_title('Recon')
            
        plt.tight_layout()
        os.makedirs('outputs/results/vae', exist_ok=True)
        plt.savefig(f'outputs/results/vae/epoch_{epoch}.png')
        plt.close()
        print(f"Saved visualization to outputs/results/vae/epoch_{epoch}.png")

def main():
    results_dir = 'outputs/results/vae'
    os.makedirs(results_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | Model: VAE (Rotation Prediction)")
    
    # 数据准备
    transform = transforms.ToTensor()
    mnist_train = datasets.MNIST('data', train=True, download=True, transform=transform)
    
    # 使用 RotatedMNIST，angle=45度，预测旋转后的图像
    train_dataset = RotatedMNIST(mnist_train, angle=45, angle_per_digit=True)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # 初始化模型 - 使用 MLP VAE 或 ResVAE
    # model = res20(latent_dim=20).to(device) # ResVAE
    model = VAE(latent_dim=20).to(device) # MLP VAE
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(1, 11):
        train(epoch, model, train_loader, optimizer, device)
        visualize_reconstruction(model, train_loader, device, epoch)

if __name__ == '__main__':
    main()
