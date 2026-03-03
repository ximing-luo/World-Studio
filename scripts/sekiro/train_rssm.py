import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import sys
import time
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, save_image

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.model.sekiro.rssm import ConvSekiroRSSM, ResNetSekiroRSSM
from src.datasets.sekiro import Sekiro_RSSM_Dataset
from src.model.components.loss import kl_divergence
from src.train.train_utils import get_log_dir

def train(epoch, model, train_loader, optimizer, device, beta=1.0, stochastic_dim=64, writer=None):
    model.train()
    total_loss = 0
    start_time = time.time()
    scaler = torch.amp.GradScaler('cuda') # 开启通用 AMP 混合精度加速
    
    for batch_idx, (seq, actions) in enumerate(train_loader):
        # seq: [B, T, C, H, W] -> [T, B, C, H, W]
        # 使用 permute + contiguous 确保在 GPU 上是连续内存，极大提升 reshape 速度
        seq = seq.permute(1, 0, 2, 3, 4).to(device, non_blocking=True).float().contiguous() / 255.0
        actions = actions.permute(1, 0, 2).to(device, non_blocking=True).float()
        
        T, B, C, H, W = seq.size()
        
        with torch.amp.autocast('cuda'): # 开启通用自动混合精度
            # --- 优化 1: 预编码 (Batch Encoding) ---
            obs_flat = seq.view(T * B, C, H, W)
            obs_feats = model.encode(obs_flat) 
            obs_feats = obs_feats.view(T, B, -1) 
            
            state = model.init_state(B, device)
            batch_recon_loss = 0
            batch_kl_loss = 0
            post_h_list = []
            post_s_list = []
            
            for t in range(T):
                obs_feat = obs_feats[t]
                current_action = torch.zeros_like(actions[t]) if t == 0 else actions[t-1]
                
                h_prev, s_prev = state
                rnn_input = torch.cat([s_prev, current_action], dim=-1)
                h_t = model.rnn_cell(rnn_input, h_prev)
                
                post_params = model.post_net(torch.cat([h_t, obs_feat], dim=-1))
                post_mu, post_logvar = torch.chunk(post_params, 2, dim=-1)
                s_t = model.reparameterize(post_mu, post_logvar)
                
                prior_params = model.prior_net(h_t)
                prior_mu, prior_logvar = torch.chunk(prior_params, 2, dim=-1)
                
                kl_loss = kl_divergence(post_mu, post_logvar, prior_mu, prior_logvar) / T
                batch_kl_loss += kl_loss
                
                state = (h_t, s_t)
                post_h_list.append(h_t)
                post_s_list.append(s_t)
            
            # --- 优化 2: 批量解码 (Batch Decoding) ---
            all_h = torch.stack(post_h_list, dim=0).view(T * B, -1)
            all_s = torch.stack(post_s_list, dim=0).view(T * B, -1)
            
            recon_obs_flat = model.decode(all_h, all_s) 
            recon_obs = recon_obs_flat.view(T, B, C, H, W)
            
            recon_loss = F.mse_loss(recon_obs, seq, reduction='sum') / (B * T)
            batch_recon_loss = recon_loss
            loss = batch_recon_loss + beta * batch_kl_loss
        
        optimizer.zero_grad()
        # AMP 的反向传播
        scaler.scale(loss).backward()
        # 混合精度下的梯度裁剪
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1000.0) 
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        
        if batch_idx % 10 == 0: # 稍微调低打印频率，减少同步开销
            pixel_mse = batch_recon_loss.item() / (C * H * W)
            rmse_255 = (pixel_mse ** 0.5) * 255
            kl_per_dim = batch_kl_loss.item() / stochastic_dim 
            elapsed = time.time() - start_time
            print(f"[{elapsed:.2f}s] Epoch {epoch} [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.0f} (Pixel MSE: {pixel_mse:.4f}/{rmse_255:.1f}, KL/Dim: {kl_per_dim:.2f}), Norm: {norm:.0f}")
            
            if writer is not None:
                step = (epoch - 1) * len(train_loader) + batch_idx
                writer.add_scalar('Loss/train', loss.item(), step)
                writer.add_scalar('ReconLoss/train', batch_recon_loss.item(), step)
                writer.add_scalar('KLLoss/train', batch_kl_loss.item(), step)
    
    avg_loss = total_loss / len(train_loader)
    print(f"====> Epoch: {epoch} Average loss: {avg_loss:.4f}")
    return avg_loss

def visualize(model, loader, device, epoch, writer=None):
    model.eval()
    num_samples = 3
    with torch.no_grad(), torch.amp.autocast('cuda'): # 开启 AMP 加速
        seq, actions = next(iter(loader))
        # 预处理数据，尽量保持 GPU 连续内存
        seq = seq.permute(1, 0, 2, 3, 4).to(device, non_blocking=True).float().contiguous() / 255.0
        actions = actions.permute(1, 0, 2).to(device, non_blocking=True).float()
        
        T, B, C, H, W = seq.size()
        B = min(B, num_samples)
        seq = seq[:, :B]
        actions = actions[:, :B]
        
        # --- 优化 1: 批量预编码 (Batch Encoding) ---
        obs_flat = seq.reshape(T * B, C, H, W)
        obs_feats = model.encode(obs_flat).view(T, B, -1)
        
        # --- 任务 1: 全程 Observe (Reconstruction) ---
        state_obs = model.init_state(B, device)
        recon_h_list = []
        recon_s_list = []
        
        for t in range(T):
            obs_feat = obs_feats[t]
            current_action = torch.zeros_like(actions[t]) if t == 0 else actions[t-1]
            
            # 使用模型内部 observe 逻辑，但避免重复调用 encode
            h_prev, s_prev = state_obs
            rnn_input = torch.cat([s_prev, current_action], dim=-1)
            h_t = model.rnn_cell(rnn_input, h_prev)
            
            post_params = model.post_net(torch.cat([h_t, obs_feat], dim=-1))
            post_mu, post_logvar = torch.chunk(post_params, 2, dim=-1)
            s_t = model.reparameterize(post_mu, post_logvar)
            
            state_obs = (h_t, s_t)
            recon_h_list.append(h_t)
            recon_s_list.append(s_t)
            
        # --- 任务 2: 部分 Observe + 部分 Imagine (Dream) ---
        state_dream = model.init_state(B, device)
        obs_len = 3
        dream_h_list = []
        dream_s_list = []
        
        # 1. Observe
        for t in range(obs_len):
            obs_feat = obs_feats[t]
            current_action = torch.zeros_like(actions[t]) if t == 0 else actions[t-1]
            
            h_prev, s_prev = state_dream
            rnn_input = torch.cat([s_prev, current_action], dim=-1)
            h_t = model.rnn_cell(rnn_input, h_prev)
            
            post_params = model.post_net(torch.cat([h_t, obs_feat], dim=-1))
            post_mu, post_logvar = torch.chunk(post_params, 2, dim=-1)
            s_t = model.reparameterize(post_mu, post_logvar)
            
            state_dream = (h_t, s_t)
            dream_h_list.append(h_t)
            dream_s_list.append(s_t)
            
        # 2. Imagine
        for t in range(obs_len, T):
            current_action = actions[t-1]
            
            h_prev, s_prev = state_dream
            rnn_input = torch.cat([s_prev, current_action], dim=-1)
            h_t = model.rnn_cell(rnn_input, h_prev)
            
            prior_params = model.prior_net(h_t)
            prior_mu, prior_logvar = torch.chunk(prior_params, 2, dim=-1)
            s_t = model.reparameterize(prior_mu, prior_logvar)
            
            state_dream = (h_t, s_t)
            dream_h_list.append(h_t)
            dream_s_list.append(s_t)
            
        # --- 优化 2: 批量解码 (Batch Decoding) ---
        # 所有的 Reconstruction 状态
        all_recon_h = torch.stack(recon_h_list, dim=0).view(T * B, -1)
        all_recon_s = torch.stack(recon_s_list, dim=0).view(T * B, -1)
        obs_recon_all = model.decode(all_recon_h, all_recon_s).view(T, B, C, H, W)
        
        # 所有的 Dream 状态
        all_dream_h = torch.stack(dream_h_list, dim=0).view(T * B, -1)
        all_dream_s = torch.stack(dream_s_list, dim=0).view(T * B, -1)
        dream_all = model.decode(all_dream_h, all_dream_s).view(T, B, C, H, W)
            
        # --- 拼接图像 ( make_grid ) ---
        all_grids = []
        for i in range(B):
            gt_row = seq[:, i] # [T, C, H, W]
            obs_row = obs_recon_all[:, i] # [T, C, H, W]
            dream_row = dream_all[:, i] # [T, C, H, W]
            
            # 拼接这三行: GT, Full Observe, Dream
            sample_grid = torch.cat([gt_row, obs_row, dream_row], dim=0) # [3T, C, H, W]
            all_grids.append(sample_grid)
            
        # 最终大图: [B*3T, C, H, W] -> 每行展示 T 张图
        final_tensor = torch.cat(all_grids, dim=0)
        grid_img = make_grid(final_tensor, nrow=T, padding=2, normalize=False)
        
        # 记录到 TensorBoard 和本地 (这里依然是 CPU 瓶颈，但已经降到了最低)
        if writer is not None:
            writer.add_image('Visual/Dream_Compare', grid_img, epoch)
            
        save_path = f'outputs/results/sekiro/rssm/dream_epoch_{epoch}.png'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        save_image(grid_img, save_path)
        print(f"Visualization saved to {save_path}")

def eval_task(model, loader, device, beta=1.0, writer=None, epoch=None):
    """
    验证任务：计算验证集上的 Loss
    """
    model.eval()
    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0
    
    with torch.no_grad():
        for seq, actions in loader:
            seq = seq.transpose(0, 1).to(device).float() / 255.0
            actions = actions.transpose(0, 1).to(device).float()
            
            T, B, C, H, W = seq.size()
            state = model.init_state(B, device)
            
            batch_recon_loss = 0
            batch_kl_loss = 0
            
            for t in range(T):
                obs = seq[t]
                current_action = torch.zeros_like(actions[t]) if t == 0 else actions[t-1]
                
                post_state, (post_mu, post_logvar) = model.observe(obs, state, current_action)
                _, (prior_mu, prior_logvar) = model.imagine(state, current_action)
                recon_obs = model.decode(*post_state)
                
                # 损失计算：除以 B*T 和 T，实现对 batch 和 序列长度 的双重平均
                recon_loss = F.mse_loss(recon_obs, obs, reduction='sum') / (B * T)
                kl_loss = kl_divergence(post_mu, post_logvar, prior_mu, prior_logvar) / T
                
                batch_recon_loss += recon_loss
                batch_kl_loss += kl_loss
                state = post_state
            
            loss = batch_recon_loss + beta * batch_kl_loss
            
            total_loss += loss.item()
            total_recon_loss += batch_recon_loss.item()
            total_kl_loss += batch_kl_loss.item()
            
    avg_loss = total_loss / len(loader)
    print(f"====> Validation Loss: {avg_loss:.4f} (Recon: {total_recon_loss/len(loader):.4f}, KL: {total_kl_loss/len(loader):.4f})")
    
    if writer is not None and epoch is not None:
        writer.add_scalar('Loss/val', avg_loss, epoch)
        writer.add_scalar('ReconLoss/val', total_recon_loss/len(loader), epoch)
        writer.add_scalar('KLLoss/val', total_kl_loss/len(loader), epoch)
        
    return avg_loss

def main():
    # Hyperparameters
    batch_size = 8
    learning_rate = 1e-4
    epochs = 30
    beta = 1.0 # KL 权重
    seq_len = 8
    stochastic_dim = 64
    deterministic_dim = 512
    action_dim = 13
    frame_skip = 3
    log_dir = get_log_dir('logs/sekiro/rssm')
    
    # DataLoader num_workers: Windows 下开启多进程会导致内存占用飙升
    num_workers = 0
    
    # TensorBoard
    writer = SummaryWriter(log_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | Task: Sekiro RSSM World Model")

    # 数据准备
    # 录制是 30fps，采样 frame_skip=3 变成 10fps，这样模型更容易学到物理变化
    train_dataset = Sekiro_RSSM_Dataset(
        data_dir='data/Sekiro/recordings', 
        seq_len=seq_len, 
        frame_skip=frame_skip
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True) # Reuse dataset for simplicity

    # 初始化模型 - 支持 Conv, ResNet 架构
    model = ResNetSekiroRSSM(stochastic_dim=stochastic_dim, deterministic_dim=deterministic_dim, action_dim=action_dim).to(device)
    # model = ConvSekiroRSSM(in_channels=3, deterministic_dim=deterministic_dim, stochastic_dim=stochastic_dim, action_dim=action_dim).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(1, epochs + 1):
        train(epoch, model, train_loader, optimizer, device, beta, stochastic_dim, writer)
        eval_task(model, test_loader, device, beta, writer, epoch)
        visualize(model, train_loader, device, epoch, writer)

    # 保存模型
    os.makedirs('outputs/models', exist_ok=True)
    torch.save(model.state_dict(), 'outputs/models/sekiro_rssm.pth')
    print(f"Model saved to 'outputs/models/sekiro_rssm.pth'")
    
    writer.close()

if __name__ == "__main__":
    main()
