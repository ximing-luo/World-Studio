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

from src.world.vision.mnist import MNISTConv, MNISTResNet
from src.world.projection.projection import LinearProjection
from src.world.latents.vae import VAELatent
from src.world.predictor.predictor import TransformerPredictor
from src.world.dream.rssm import TemporalGenerative
from src.datasets.mnist import MNIST_RSSM_Dataset

def kl_divergence(mu1, logvar1, mu2, logvar2):
    """计算两个高斯分布之间的 KL 散度: KL(N(mu1, sigma1) || N(mu2, sigma2))"""
    var1 = torch.exp(logvar1)
    var2 = torch.exp(logvar2)
    kl = 0.5 * (logvar2 - logvar1 + (var1 + (mu1 - mu2)**2) / var2 - 1.0)
    return kl.sum(dim=-1).mean()

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 数据准备
    mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
    # 序列长度为 8，每步旋转 15 度
    train_dataset = MNIST_RSSM_Dataset(mnist_train, seq_len=8, angle=15)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)

    # 框架化构建模型
    latent_dim = 32
    vision = MNISTConv(in_channels=1)
    projection = LinearProjection(in_channels=128, height=7, width=7, token_dim=latent_dim, is_vae=True)
    latent = VAELatent()
    # 预测器：输入潜变量，输出预测潜变量参数 (mu, logvar)
    # 这里的输入维度是 latent_dim，输出维度是 latent_dim * 2 (用于 mu/logvar)
    predictor = TransformerPredictor(input_dim=latent_dim, output_dim=latent_dim * 2, hidden_dim=128, num_layers=4)
    
    model = TemporalGenerative(vision, projection, latent, predictor).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    epochs = 10
    beta = 1.0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        total_recon = 0
        total_kl = 0
        
        for batch_idx, (seq, actions) in enumerate(train_loader):
            # seq: [B, T, C, H, W]
            seq = seq.to(device).float()
            # actions: [B, T, 1]
            actions = actions.to(device).float()
            
            optimizer.zero_grad()
            
            # 使用框架的 observe 逻辑
            # 返回: 先验参数 (B, T*S, 2D), 后验采样 z (B, T*S, D), 隐空间损失 (KL 与先验对齐)
            prior_params, z_post, latent_loss = model.observe(seq)
            
            # 批量解码
            # z_post: (B, T*S, D) -> 需要调整为 (B*T, S, D) 以便批量解码
            B, T_S, D = z_post.shape
            S = model.projection.num_tokens
            T = T_S // S
            
            z_post_reshaped = z_post.view(B * T, S, D)
            recon_seq = model.decode(z_post_reshaped) 
            recon_seq = recon_seq.view(B, T, *seq.shape[2:])
            
            # 重建损失
            recon_loss = F.mse_loss(recon_seq, seq, reduction='sum') / B
            
            # 总损失
            loss = recon_loss + beta * latent_loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += latent_loss.item()
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch} [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f} (Recon: {recon_loss.item():.4f}, KL: {latent_loss.item():.4f})")
        
        avg_loss = total_loss / len(train_loader)
        print(f"====> Epoch: {epoch} Average loss: {avg_loss:.4f}")
        
        # 可视化
        visualize_dream(model, train_loader, device, epoch)

    # 保存模型
    os.makedirs('outputs/models', exist_ok=True)
    torch.save(model.state_dict(), 'outputs/models/mnist_rssm.pth')
    print(f"Model saved to 'outputs/models/mnist_rssm.pth'")

def visualize_dream(model, loader, device, epoch, num_samples=5):
    model.eval()
    with torch.no_grad():
        seq, actions = next(iter(loader))
        seq = seq.to(device).float()
        
        B, T, C, H, W = seq.size()
        B = min(B, num_samples)
        seq = seq[:B]
        
        # 1. 观察前 3 帧
        obs_len = 3
        obs_seq = seq[:, :obs_len]
        
        # 获取前 3 帧的后验分布
        _, z_post_obs, _ = model.observe(obs_seq) # (B, obs_len * S, D)
        
        # 2. 想象后续帧
        # 从 z_post_obs 开始推演
        current_z_seq = z_post_obs
        all_recon = []
        
        # 解码已观察部分
        recon_obs = model.decode(z_post_obs) # (B * obs_len, C, H, W)
        recon_obs = recon_obs.view(B, obs_len, C, H, W)
        for t in range(obs_len):
            all_recon.append(recon_obs[:, t])
            
        # 逐步推演
        for t in range(obs_len, T):
            # 想象下一帧
            z_next, _ = model.imagine_next(current_z_seq) # (B, S, D)
            
            # 解码下一帧
            recon_next = model.decode(z_next) # (B, C, H, W)
            all_recon.append(recon_next)
            
            # 更新历史潜变量序列
            current_z_seq = torch.cat([current_z_seq, z_next], dim=1)
            
        # 绘图
        fig, axes = plt.subplots(B, T, figsize=(T*2, B*2))
        if B == 1: axes = axes.reshape(1, -1)
        
        for i in range(B):
            for t in range(T):
                ax = axes[i, t]
                img = all_recon[t][i].cpu().squeeze().numpy()
                ax.imshow(img, cmap='gray')
                ax.axis('off')
                if t < obs_len:
                    ax.set_title("Observed" if i==0 else "")
                else:
                    ax.set_title("Imagined" if i==0 else "")
        
        plt.tight_layout()
        os.makedirs('outputs/results/mnist/rssm', exist_ok=True)
        plt.savefig(f'outputs/results/mnist/rssm/dream_epoch_{epoch}.png')
        plt.close()
        print(f"Saved visualization to outputs/results/mnist/rssm/dream_epoch_{epoch}.png")

if __name__ == "__main__":
    train()
