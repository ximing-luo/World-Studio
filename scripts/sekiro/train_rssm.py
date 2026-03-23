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

from src.world.vision.sekiro import SekiroConv, SekiroResNet
from src.world.projection.projection import SpatialProjection
from src.world.latents.vae import VAELatent
from src.world.predictor.predictor import SpatioTemporalPredictor
from src.world.dream.rssm import TemporalGenerative
from src.datasets.sekiro import Sekiro_RSSM_Dataset
from src.utils.loss import kl_divergence
from src.utils.train_utils import get_log_dir

def train(epoch, model, train_loader, optimizer, device, beta=1.0, writer=None):
    model.train()
    total_loss = 0
    start_time = time.time()
    scaler = torch.amp.GradScaler('cuda') # 开启通用 AMP 混合精度加速
    
    for batch_idx, (seq, actions) in enumerate(train_loader):
        # seq: [B, T, C, H, W]
        seq = seq.to(device, non_blocking=True).float() / 255.0
        # actions: [B, T, D]
        actions = actions.to(device, non_blocking=True).float()
        
        T, B, C, H, W = seq.size(1), seq.size(0), seq.size(2), seq.size(3), seq.size(4)
        
        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            # 使用框架的 observe 逻辑
            # 返回: 先验参数, 后验采样 z, 隐空间损失
            prior_params, z_post, latent_loss = model.observe(seq)
            
            # 批量解码所有后验状态
            # z_post: (B, T*S, D) -> 需要调整为 (B*T, S, D) 以便批量解码
            B, T_S, D = z_post.shape
            S = model.projection.num_tokens
            T = T_S // S
            
            z_post_reshaped = z_post.view(B * T, S, D)
            recon_seq = model.decode(z_post_reshaped) 
            recon_seq = recon_seq.view(B, T, C, H, W)
            
            recon_loss = F.mse_loss(recon_seq, seq, reduction='sum') / (B * T)
            loss = recon_loss + beta * latent_loss
        
        # AMP 的反向传播
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0) 
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        
        if batch_idx % 10 == 0:
            elapsed = time.time() - start_time
            print(f"[{elapsed:.2f}s] Epoch {epoch} [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f} (Rec: {recon_loss.item():.4f}, KL: {latent_loss.item():.4f})")
            
            if writer is not None:
                step = (epoch - 1) * len(train_loader) + batch_idx
                writer.add_scalar('Loss/train', loss.item(), step)
    
    avg_loss = total_loss / len(train_loader)
    return avg_loss

def visualize(model, loader, device, epoch, writer=None):
    model.eval()
    num_samples = 3
    with torch.no_grad(), torch.amp.autocast('cuda'):
        seq, actions = next(iter(loader))
        seq = seq.to(device, non_blocking=True).float() / 255.0
        
        B, T, C, H, W = seq.size()
        B = min(B, num_samples)
        seq = seq[:B]
        
        # 1. 观察前 3 帧
        obs_len = 3
        obs_seq = seq[:, :obs_len]
        
        # 获取前 3 帧的后验分布
        _, z_post_obs, _ = model.observe(obs_seq)
        
        # 2. 想象后续帧
        current_z_seq = z_post_obs
        all_recon = []
        
        # 解码已观察部分
        recon_obs = model.decode(z_post_obs)
        recon_obs = recon_obs.view(B, obs_len, C, H, W)
        for t in range(obs_len):
            all_recon.append(recon_obs[:, t])
            
        # 逐步推演
        for t in range(obs_len, T):
            z_next, _ = model.imagine_next(current_z_seq)
            recon_next = model.decode(z_next)
            all_recon.append(recon_next)
            current_z_seq = torch.cat([current_z_seq, z_next], dim=1)
            
        # 拼接图像
        all_grids = []
        for i in range(B):
            gt_row = seq[i] # [T, C, H, W]
            recon_row = torch.stack([all_recon[t][i] for t in range(T)], dim=0) # [T, C, H, W]
            
            # 拼接 GT 和 Recon
            sample_grid = torch.cat([gt_row, recon_row], dim=0) # [2T, C, H, W]
            all_grids.append(sample_grid)
            
        final_tensor = torch.cat(all_grids, dim=0)
        grid_img = make_grid(final_tensor, nrow=T, padding=2, normalize=False)
        
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
    
    with torch.no_grad():
        for seq, actions in loader:
            seq = seq.to(device).float() / 255.0
            
            # 使用框架的 observe 逻辑
            _, _, latent_loss = model.observe(seq)
            
            # 简化验证集损失计算
            total_loss += latent_loss.item()
                
    avg_loss = total_loss / len(loader)
    print(f"====> Validation KL Loss: {avg_loss:.4f}")
    
    if writer is not None and epoch is not None:
        writer.add_scalar('Loss/val_kl', avg_loss, epoch)
        
    return avg_loss

def main():
    # Hyperparameters
    batch_size = 4
    learning_rate = 1e-4
    epochs = 30
    beta = 1.0
    seq_len = 4
    latent_dim = 64
    action_dim = 13
    frame_skip = 3
    log_dir = get_log_dir('logs/sekiro/rssm')
    
    num_workers = 0
    writer = SummaryWriter(log_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | Task: Sekiro RSSM World Model")

    # 数据准备
    train_dataset = Sekiro_RSSM_Dataset(
        data_dir='data/Sekiro/recordings', 
        seq_len=seq_len, 
        frame_skip=frame_skip
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # 框架化构建模型
    vision = SekiroConv()
    # SekiroConv 输出通道 512，H=4, W=7 (128x240 -> 4x7)
    projection = SpatialProjection(512, latent_dim, 4, 7, is_vae=True)
    latent = VAELatent()
    # 预测器：时空预测器
    predictor = SpatioTemporalPredictor(input_dim=latent_dim, output_dim=latent_dim * 2, hidden_dim=256, height=4, width=7)
    
    model = TemporalGenerative(vision, projection, latent, predictor).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(1, epochs + 1):
        train(epoch, model, train_loader, optimizer, device, beta, writer)
        eval_task(model, test_loader, device, beta, writer, epoch)
        visualize(model, train_loader, device, epoch, writer)

    # 保存模型
    os.makedirs('outputs/models', exist_ok=True)
    torch.save(model.state_dict(), 'outputs/models/sekiro_rssm.pth')
    print(f"Model saved to 'outputs/models/sekiro_rssm.pth'")
    
    writer.close()

if __name__ == "__main__":
    main()
