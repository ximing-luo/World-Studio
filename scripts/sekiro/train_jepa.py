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

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.model.sekiro.jepa import ConvSekiroJEPA, ResNetSekiroJEPA
from src.datasets.sekiro import Sekiro_JEPA_Dataset

def validate_action_regression(model, device, test_loader):
    """
    验证任务：动作回归 (Action Regression)
    由于 Sekiro 的动作空间 [1, 2] 是连续/多维的，这里简单计算特征预测的误差。
    """
    model.eval()
    total_error = 0
    count = 0
    
    with torch.no_grad():
        for i, (img, next_img, action) in enumerate(test_loader):
            if i > 10: break
            img, next_img, action = img.to(device), next_img.to(device), action.to(device)
            
            z_context = model.context_encoder(img)
            z_target = model.target_encoder(next_img)
            
            # 预测特征
            z_target_pred = model.predictor(torch.cat([z_context, action], dim=-1))
            
            # 计算特征预测的 L2 距离
            error = torch.norm(z_target_pred - z_target, dim=1).mean()
            total_error += error.item()
            count += 1
                
    return total_error / count if count > 0 else 0

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | Task: Sekiro JEPA World Model")

    # 数据准备
    dataset = Sekiro_JEPA_Dataset(data_dir='data/demos')
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    # 模型初始化 - 支持 Conv, ResNet 架构
    # model = ResNetSekiroJEPA(in_channels=3, latent_dim=256, action_dim=15).to(device)
    model = ConvSekiroJEPA(in_channels=3, latent_dim=256, action_dim=15).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    epochs = 10
    history = {
        'jepa_loss': [],
        'feature_error': []
    }
    
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_jepa_loss = 0
        
        for batch_idx, (img, next_img, action) in enumerate(train_loader):
            img, next_img, action = img.to(device), next_img.to(device), action.to(device)
            
            # --- JEPA 自监督训练 ---
            optimizer.zero_grad()
            _, _, jepa_loss = model(img, next_img, action)
            jepa_loss.backward()
            optimizer.step()
            model.update_target(momentum=0.99)
            
            total_jepa_loss += jepa_loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch} [{batch_idx}/{len(train_loader)}] Loss: {jepa_loss.item():.4f}")

        # 验证
        feat_error = validate_action_regression(model, device, test_loader)
        history['jepa_loss'].append(total_jepa_loss / len(train_loader))
        history['feature_error'].append(feat_error)
        
        print(f"====> Epoch: {epoch} Avg JEPA Loss: {history['jepa_loss'][-1]:.4f} | Feature Error: {feat_error:.4f}")

    # 保存模型
    os.makedirs('outputs/models', exist_ok=True)
    torch.save(model.state_dict(), 'outputs/models/sekiro_jepa.pth')
    print(f"Model saved to 'outputs/models/sekiro_jepa.pth'")

    # 可视化结果
    save_plots(history)
    print(f"Training finished in {time.time() - start_time:.1f}s. Plots saved to 'outputs/results/sekiro/jepa/jepa_results.png'")

def save_plots(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 1. JEPA Loss
    axes[0].plot(history['jepa_loss'], color='blue', label='JEPA Loss')
    axes[0].set_title('Training Loss')
    axes[0].grid(True)
    
    # 2. Feature Error
    axes[1].plot(history['feature_error'], color='red', label='L2 Feature Error')
    axes[1].set_title('Next-Frame Feature Prediction Error')
    axes[1].grid(True)
    
    plt.tight_layout()
    os.makedirs('outputs/results/sekiro/jepa', exist_ok=True)
    plt.savefig('outputs/results/sekiro/jepa/jepa_results.png')
    plt.close()

if __name__ == "__main__":
    train()
