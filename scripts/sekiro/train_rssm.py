import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.model.sekiro.rssm import ConvSekiroRSSM, ResNetSekiroRSSM
from src.datasets.sekiro import Sekiro_RSSM_Dataset

def kl_divergence(mu1, logvar1, mu2, logvar2):
    var1 = torch.exp(logvar1)
    var2 = torch.exp(logvar2)
    kl = 0.5 * (logvar2 - logvar1 + (var1 + (mu1 - mu2)**2) / var2 - 1.0)
    return kl.sum(dim=-1).mean()

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | Task: Sekiro RSSM World Model")

    # 数据准备
    # 录制是 30fps，采样 frame_skip=3 变成 10fps，这样模型更容易学到物理变化
    train_dataset = Sekiro_RSSM_Dataset(data_dir='data/demos', seq_len=16, frame_skip=3)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    # 初始化模型 - 支持 Conv, ResNet 架构
    # model = ResNetSekiroRSSM(latent_dim=256).to(device)
    # Action dim = 15 (只狼的动作数据维度)
    model = ConvSekiroRSSM(in_channels=3, deterministic_dim=512, stochastic_dim=64, action_dim=15).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    epochs = 30
    beta = 1.0 # KL 权重

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        
        for batch_idx, (seq, actions) in enumerate(train_loader):
            # seq: [B, T, C, H, W] -> [T, B, C, H, W]
            seq = seq.transpose(0, 1).to(device).float()
            # actions: [B, T, 2] -> [T, B, 2]
            actions = actions.transpose(0, 1).to(device).float()
            
            T, B, C, H, W = seq.size()
            state = model.init_state(B, device)
            
            batch_recon_loss = 0
            batch_kl_loss = 0
            
            for t in range(T):
                obs = seq[t]
                # 在第 t 步，我们使用 action[t-1] 产生的效果
                if t == 0:
                    current_action = torch.zeros_like(actions[t])
                else:
                    current_action = actions[t-1]
                
                # 1. Observe (Posterior)
                post_state, (post_mu, post_logvar) = model.observe(obs, state, current_action)
                # 2. Imagine (Prior)
                _, (prior_mu, prior_logvar) = model.imagine(state, current_action)
                # 3. Decode
                recon_obs = model.decode(*post_state)
                
                # Loss
                recon_loss = F.mse_loss(recon_obs, obs, reduction='sum') / B
                kl_loss = kl_divergence(post_mu, post_logvar, prior_mu, prior_logvar)
                
                batch_recon_loss += recon_loss
                batch_kl_loss += kl_loss
                state = post_state
            
            loss = batch_recon_loss + beta * batch_kl_loss
            
            optimizer.zero_grad()
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10000.0)
            print(f"Grad norm: {norm:.4f}")
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 5 == 0:
                print(f"Epoch {epoch} [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f} (Recon: {batch_recon_loss.item():.4f}, KL: {batch_kl_loss.item():.4f})")
        
        print(f"====> Epoch: {epoch} Average loss: {total_loss / len(train_loader):.4f}")
        visualize_dream(model, train_loader, device, epoch)

    # 保存模型
    os.makedirs('outputs/models', exist_ok=True)
    torch.save(model.state_dict(), 'outputs/models/sekiro_rssm.pth')
    print(f"Model saved to 'outputs/models/sekiro_rssm.pth'")

def visualize_dream(model, loader, device, epoch, num_samples=3):
    model.eval()
    with torch.no_grad():
        seq, actions = next(iter(loader))
        seq = seq.transpose(0, 1).to(device).float()
        actions = actions.transpose(0, 1).to(device).float()
        
        T, B, C, H, W = seq.size()
        B = min(B, num_samples)
        seq = seq[:, :B]
        actions = actions[:, :B]
        
        # --- 任务 1: 全程 Observe (Reconstruction) ---
        state_obs = model.init_state(B, device)
        obs_recon_frames = []
        for t in range(T):
            obs = seq[t]
            current_action = torch.zeros_like(actions[t]) if t == 0 else actions[t-1]
            post_state, _ = model.observe(obs, state_obs, current_action)
            recon = model.decode(*post_state)
            obs_recon_frames.append(recon.cpu())
            state_obs = post_state
            
        # --- 任务 2: 部分 Observe + 部分 Imagine (Dream) ---
        state_dream = model.init_state(B, device)
        obs_len = 3
        dream_frames = []
        
        # 1. Observe
        for t in range(obs_len):
            obs = seq[t]
            current_action = torch.zeros_like(actions[t]) if t == 0 else actions[t-1]
            post_state, _ = model.observe(obs, state_dream, current_action)
            recon = model.decode(*post_state)
            dream_frames.append(recon.cpu())
            state_dream = post_state
            
        # 2. Imagine
        for t in range(obs_len, T):
            current_action = actions[t-1]
            prior_state, _ = model.imagine(state_dream, current_action)
            recon = model.decode(*prior_state)
            dream_frames.append(recon.cpu())
            state_dream = prior_state
            
        # Plot
        # 每一组 sample 展示 3 行: Ground Truth, Full Observe, Dream
        fig, axes = plt.subplots(B * 3, T, figsize=(T*2, B*6))
        
        for i in range(B):
            for t in range(T):
                # Row 1: Ground Truth
                ax_gt = axes[i*3, t]
                img_gt = seq[t][i].cpu().permute(1, 2, 0).numpy()
                ax_gt.imshow(img_gt)
                ax_gt.axis('off')
                if t == 0: ax_gt.set_ylabel(f"Sample {i}\nGT", rotation=0, labelpad=40, fontsize=12)
                if i == 0 and t == 0: ax_gt.set_title("Ground Truth")

                # Row 2: Full Observe (Reconstruction)
                ax_obs = axes[i*3 + 1, t]
                img_obs = obs_recon_frames[t][i].permute(1, 2, 0).numpy()
                ax_obs.imshow(img_obs)
                ax_obs.axis('off')
                if t == 0: ax_obs.set_ylabel("Observe", rotation=0, labelpad=40, fontsize=12)
                if i == 0 and t == 0: ax_obs.set_title("Full Observation")

                # Row 3: Dream (3 Observe + Imagine)
                ax_dream = axes[i*3 + 2, t]
                img_dream = dream_frames[t][i].permute(1, 2, 0).numpy()
                ax_dream.imshow(img_dream)
                ax_dream.axis('off')
                if t == 0: ax_dream.set_ylabel("Dream", rotation=0, labelpad=40, fontsize=12)
                if i == 0:
                    if t < obs_len:
                        ax_dream.set_title("Observed")
                    else:
                        ax_dream.set_title("Imagined")
        
        plt.tight_layout()
        os.makedirs('outputs/results/sekiro/rssm', exist_ok=True)
        plt.savefig(f'outputs/results/sekiro/rssm/dream_epoch_{epoch}.png')
        plt.close()

if __name__ == "__main__":
    train()
