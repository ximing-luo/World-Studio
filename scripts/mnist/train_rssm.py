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

from src.model.mnist.rssm import FCRSSM, ConvRSSM, ResNetRSSM
from src.datasets.mnist import MNIST_RSSM_Dataset

def kl_divergence(mu1, logvar1, mu2, logvar2):
    """计算两个高斯分布之间的 KL 散度: KL(N(mu1, sigma1) || N(mu2, sigma2))"""
    # logvar = 2 * log(sigma) -> sigma^2 = exp(logvar)
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
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)

    # 模型初始化 - 支持 FC, Conv, ResNet 三种架构
    # model = FCRSSM(action_dim=1).to(device)
    # model = ResNetRSSM(action_dim=1).to(device)
    model = ConvRSSM(action_dim=1).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    epochs = 10
    beta = 1.0 # KL 权重

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        total_recon = 0
        total_kl = 0
        
        for batch_idx, (seq, actions) in enumerate(train_loader):
            # seq: [B, T, C, H, W] -> [T, B, C, H, W]
            seq = seq.transpose(0, 1).to(device).float()
            # actions: [B, T, 1] -> [T, B, 1]
            actions = actions.transpose(0, 1).to(device).float()
            
            T, B, C, H, W = seq.size()
            
            # 初始状态
            state = model.init_state(B, device)
            
            batch_recon_loss = 0
            batch_kl_loss = 0
            
            # 遍历序列
            for t in range(T):
                obs = seq[t]
                action = actions[t]
                
                # Action Logic:
                # at step t, we observe obs[t].
                # The transition to obs[t] was caused by action[t-1] (applied to obs[t-1]).
                # For t=0, there is no previous action, so we use zero.
                if t == 0:
                    current_action = torch.zeros_like(action)
                else:
                    current_action = actions[t-1]
                
                # 1. 观察步 (Posterior)
                post_state, (post_mu, post_logvar) = model.observe(obs, state, current_action)
                
                # 2. 想象步 (Prior) - 用于计算 KL 散度，让先验向后验靠拢
                _, (prior_mu, prior_logvar) = model.imagine(state, current_action)
                
                # 3. 解码 (从后验状态还原为图像)
                recon_obs = model.decode_state(post_state[0], post_state[1])
                
                # 计算损失
                recon_loss = F.mse_loss(recon_obs, obs, reduction='sum') / B
                kl_loss = kl_divergence(post_mu, post_logvar, prior_mu, prior_logvar)
                
                batch_recon_loss += recon_loss
                batch_kl_loss += kl_loss
                
                # 更新状态为当前后验状态，用于下一步
                state = post_state
            
            loss = batch_recon_loss + beta * batch_kl_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_recon += batch_recon_loss.item()
            total_kl += batch_kl_loss.item()
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch} [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f} (Recon: {batch_recon_loss.item():.4f}, KL: {batch_kl_loss.item():.4f})")
        
        avg_loss = total_loss / len(train_loader)
        print(f"====> Epoch: {epoch} Average loss: {avg_loss:.4f}")
        
        # 每个 epoch 结束后进行一次可视化
        visualize_dream(model, train_loader, device, epoch)

    # 保存模型
    os.makedirs('outputs/models', exist_ok=True)
    torch.save(model.state_dict(), 'outputs/models/mnist_rssm.pth')
    print(f"Model saved to 'outputs/models/mnist_rssm.pth'")

def visualize_dream(model, loader, device, epoch, num_samples=5):
    model.eval()
    with torch.no_grad():
        seq, actions = next(iter(loader))
        seq = seq.transpose(0, 1).to(device).float()
        actions = actions.transpose(0, 1).to(device).float()
        
        T, B, C, H, W = seq.size()
        B = min(B, num_samples)
        
        # 只取前几个样本
        seq = seq[:, :B]
        actions = actions[:, :B]
        
        state = model.init_state(B, device)
        
        # 观察前 3 帧
        obs_len = 3
        recon_frames = []
        
        # 1. Observe Phase (Posterior)
        for t in range(obs_len):
            obs = seq[t]
            if t == 0:
                current_action = torch.zeros_like(actions[t])
            else:
                current_action = actions[t-1]
                
            post_state, _ = model.observe(obs, state, current_action)
            recon = model.decode_state(post_state[0], post_state[1])
            recon_frames.append(recon.cpu())
            state = post_state
            
        # 2. Imagine Phase (Prior)
        # To predict frame t, we use state_{t-1} and action_{t-1}
        for t in range(obs_len, T):
            current_action = actions[t-1]
            
            prior_state, _ = model.imagine(state, current_action)
            recon = model.decode_state(prior_state[0], prior_state[1])
            recon_frames.append(recon.cpu())
            state = prior_state
            
        # 绘图
        fig, axes = plt.subplots(B, T, figsize=(T*2, B*2))
        if B == 1: axes = axes.reshape(1, -1)
        
        for i in range(B):
            for t in range(T):
                ax = axes[i, t]
                img = recon_frames[t][i].squeeze().numpy()
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
