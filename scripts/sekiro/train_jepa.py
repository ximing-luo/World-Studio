import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import time
from torch.utils.tensorboard import SummaryWriter

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.world.vision.sekiro import SekiroConv, SekiroResNet
from src.world.projection.projection import SpatialProjection
from src.world.latents.vicreg import VicRegLatent
from src.world.predictor.predictor import SpatioTemporalPredictor
from src.world.dream.jepa import TemporalPredictive
from src.datasets.sekiro import Sekiro_JEPA_Dataset
from src.utils.train_utils import get_log_dir

def train(epoch, model, train_loader, optimizer, device, writer=None):
    model.train()
    total_jepa_loss = 0
    start_time = time.time()
    
    for batch_idx, (img, next_img, action) in enumerate(train_loader):
        # 将 uint8 数据搬运到 GPU 后再进行浮点转换和归一化 (极致性能)
        img = img.to(device).float() / 255.0
        next_img = next_img.to(device).float() / 255.0
        action = action.to(device)
        B = img.size(0)
        
        # --- JEPA 自监督训练 ---
        optimizer.zero_grad()
        
        # 适配框架序列输入: (B, 1, C, H, W)
        _, _, jepa_loss = model(img.unsqueeze(1), next_img.unsqueeze(1), action)
        
        jepa_loss.backward()
        optimizer.step()
        model.update_target(momentum=0.99)
        
        total_jepa_loss += jepa_loss.item()
        
        if batch_idx % 10 == 0:
            elapsed = time.time() - start_time
            print(f"[{elapsed:.2f}s] Epoch {epoch} [{batch_idx * B}/{len(train_loader.dataset)}] Loss: {jepa_loss.item():.4f}")
            if writer is not None:
                step = (epoch - 1) * len(train_loader) + batch_idx
                writer.add_scalar('train/loss', jepa_loss.item(), step)
            
    avg_loss = total_jepa_loss / len(train_loader)
    return avg_loss

def visualize(model, loader, device, epoch, history=None, writer=None):
    """
    可视化训练曲线 (JEPA 没有直接的像素重建)
    """
    if history is None:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 1. JEPA Loss
    if 'jepa_loss' in history and len(history['jepa_loss']) > 0:
        axes[0].plot(range(1, len(history['jepa_loss']) + 1), history['jepa_loss'], color='blue', label='JEPA Loss')
        axes[0].set_title('Training Loss')
        axes[0].grid(True)
        axes[0].set_xlabel('Epoch')
    
    # 2. Feature Error
    if 'feature_error' in history and len(history['feature_error']) > 0:
        axes[1].plot(range(1, len(history['feature_error']) + 1), history['feature_error'], color='red', label='L2 Feature Error')
        axes[1].set_title('Next-Frame Feature Prediction Error')
        axes[1].grid(True)
        axes[1].set_xlabel('Epoch')
    
    plt.tight_layout()
    os.makedirs('outputs/results/sekiro/jepa', exist_ok=True)
    plt.savefig('outputs/results/sekiro/jepa/jepa_results.png')
    plt.close()
    print(f"Saved visualization to outputs/results/sekiro/jepa/jepa_results.png")

def eval_task(model, loader, device, writer=None, epoch=None):
    """
    验证任务：动作回归/特征预测误差
    由于 Sekiro 的动作空间 [1, 2] 是连续/多维的，这里简单计算特征预测的误差。
    """
    model.eval()
    total_error = 0
    count = 0
    
    with torch.no_grad():
        for i, (img, next_img, action) in enumerate(loader):
            if i > 10: break # 仅验证部分 batch 以节省时间
            # 将 uint8 数据搬运到 GPU 后再进行浮点转换和归一化 (极致性能)
            img = img.to(device).float() / 255.0
            next_img = next_img.to(device).float() / 255.0
            action = action.to(device)
            
            feat_c = model.vision.encode(img)
            z_context = model.projection.encode(feat_c) # (B, S, D)
            
            feat_t = model.target_vision.encode(next_img)
            z_target = model.projection.encode(feat_t) # (B, S, D)
            
            # 预测特征
            # 使用 TemporalPredictive 的条件注入逻辑
            # 这里 z_context 是 (B, S, D), action 是 (B, D_cond)
            # 按照 TemporalPredictive.forward 中的逻辑进行拼接
            B, S, D = z_context.shape
            cond = action.unsqueeze(1).expand(-1, S, -1)
            z_input = torch.cat([z_context, cond], dim=-1)
            z_target_pred = model.predictor(z_input)
            
            # 计算特征预测的 L2 距离
            error = torch.norm(z_target_pred - z_target, dim=-1).mean()
            total_error += error.item()
            count += 1
                
    avg_error = total_error / count if count > 0 else 0
    
    if writer is not None and epoch is not None:
        writer.add_scalar('val/feature_error', avg_error, epoch)
        
    return avg_error

def main():
    # Hyperparameters
    batch_size = 32
    learning_rate = 1e-4
    epochs = 10
    latent_dim = 256
    action_dim = 13
    prediction_offset = 10 # 预测未来第几帧
    log_dir = get_log_dir('logs/sekiro/jepa')
    
    # DataLoader num_workers: Windows 下开启多进程会导致内存占用飙升
    num_workers = 0 
    
    # TensorBoard
    writer = SummaryWriter(log_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | Task: Sekiro JEPA World Model")

    # 数据准备
    dataset = Sekiro_JEPA_Dataset(data_dir='data/Sekiro/recordings', frame_skip=prediction_offset)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # 框架化构建模型
    vision = SekiroConv()
    # SekiroConv 输出通道 512，H=4, W=7 (128x240 -> 4x7)
    projection = SpatialProjection(512, latent_dim, 4, 7, is_vae=False)
    latent = VicRegLatent()
    # 预测器：输入潜变量 + 动作编码
    # 按照 TemporalPredictive 的逻辑，输入维度是 latent_dim + action_dim
    predictor = SpatioTemporalPredictor(input_dim=latent_dim + action_dim, output_dim=latent_dim, hidden_dim=256, height=4, width=7)
    
    model = TemporalPredictive(vision, projection, latent, predictor).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    history = {
        'jepa_loss': [],
        'feature_error': []
    }
    
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        train_loss = train(epoch, model, train_loader, optimizer, device, writer)
        feat_error = eval_task(model, test_loader, device, writer, epoch)
        
        history['jepa_loss'].append(train_loss)
        history['feature_error'].append(feat_error)
        
        print(f"====> Epoch: {epoch} Avg JEPA Loss: {train_loss:.4f} | Feature Error: {feat_error:.4f}")
        
        visualize(model, train_loader, device, epoch, history=history, writer=writer)

    # 保存模型
    os.makedirs('outputs/models', exist_ok=True)
    torch.save(model.state_dict(), 'outputs/models/sekiro_jepa.pth')
    print(f"Model saved to 'outputs/models/sekiro_jepa.pth'")
    print(f"Training finished in {time.time() - start_time:.1f}s.")
    
    writer.close()

if __name__ == "__main__":
    main()
