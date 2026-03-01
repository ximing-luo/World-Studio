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
import time

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.model.mnist.jepa import FCJEPA, ConvJEPA, ResNetJEPA
from src.datasets.mnist import MNIST_JEPA_Dataset

def validate_semantic_classification(model, device, train_loader_probe, test_loader_probe, num_epochs=1):
    """
    验证任务零：语义分类 (Semantic Classification)
    线性探测：彻底冻结特征，重新训练一个分类器，看特征的语义潜力。
    """
    model.eval()
    
    # 动态获取隐空间维度，避免硬编码 128 导致的维度不匹配错误
    latent_dim = model.context_encoder[-1].out_features
    classifier = nn.Linear(latent_dim, 10).to(device)
    
    optimizer = optim.Adam(classifier.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0
    for _ in range(num_epochs):
        classifier.train()
        for data, labels in train_loader_probe:
            data, labels = data.to(device), labels.to(device)
            with torch.no_grad():
                z = model.context_encoder(data).detach()
            
            optimizer.zero_grad()
            outputs = classifier(z)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
        classifier.eval()
        correct = 0
        with torch.no_grad():
            for data, labels in test_loader_probe:
                data, labels = data.to(device), labels.to(device)
                z = model.context_encoder(data)
                pred = classifier(z).argmax(dim=1)
                correct += pred.eq(labels).sum().item()
        
        acc = 100. * correct / len(test_loader_probe.dataset)
        best_acc = max(best_acc, acc)
        
    return best_acc

def validate_action_regression(model, device, test_loader):
    """
    验证任务一：动作回归 (Action Regression)
    寻找 a* = argmin_a ||z_pred(x, a) - z_target(x')||
    """
    model.eval()
    total_error = 0
    count = 0
    
    # 候选角度：0-355度，步长5度
    candidate_angles = np.arange(0, 360, 5)
    candidate_rads = torch.tensor(candidate_angles / 180.0 * 3.14159, dtype=torch.float32).to(device)
    
    # 只取一部分测试集进行验证以节省时间
    with torch.no_grad():
        for i, (img, rotated_img, _, true_rad) in enumerate(test_loader):
            if i > 20: break # 仅测试前 20 个 Batch
            
            img, rotated_img = img.to(device), rotated_img.to(device)
            B = img.size(0)
            
            # 1. 提取上下文和目标特征
            z_context = model.context_encoder(img)
            z_target = model.target_encoder(rotated_img)
            
            for b in range(B):
                # 2. 对每个样本，尝试所有候选角度
                z_c = z_context[b].unsqueeze(0).expand(len(candidate_rads), -1)
                
                # 同样进行 sin/cos 编码
                cond_rads = candidate_rads.unsqueeze(1)
                cond_emb = torch.cat([torch.sin(cond_rads), torch.cos(cond_rads)], dim=-1)
                
                # 预测特征
                z_preds = model.predictor(torch.cat([z_c, cond_emb], dim=-1))
                
                # 计算与真实目标特征的距离
                distances = torch.norm(z_preds - z_target[b].unsqueeze(0), dim=1)
                best_idx = torch.argmin(distances)
                pred_angle = candidate_angles[best_idx]
                
                # 计算真实角度 (转换为 0-360 度)
                true_angle = (true_rad[b].item() * 180.0 / 3.14159) % 360
                
                # 计算最小旋转误差 (处理 0/360 度临界点)
                error = min(abs(pred_angle - true_angle), 360 - abs(pred_angle - true_angle))
                total_error += error
                count += 1
                
    return total_error / count if count > 0 else 0

def validate_compositionality(model, device, test_loader):
    """
    验证任务二：复合一致性 (Action Compositionality)
    验证 z(x, a1+a2) ≈ z(z(x, a1), a2)
    """
    model.eval()
    total_sim = 0
    count = 0
    
    with torch.no_grad():
        for i, (img, _, _, _) in enumerate(test_loader):
            if i > 20: break
            
            img = img.to(device)
            B = img.size(0)
            
            # 随机生成两个动作 a1, a2 (弧度)
            a1 = torch.rand(B, 1).to(device) * 2 * 3.14159
            a2 = torch.rand(B, 1).to(device) * 2 * 3.14159
            
            z_base = model.context_encoder(img)
                    
            # 路径 A：一步到位 z(x, a1+a2)
            a_sum = (a1 + a2) % (2 * 3.14159)
            emb_sum = torch.cat([torch.sin(a_sum), torch.cos(a_sum)], dim=-1)
            z_single = model.predictor(torch.cat([z_base, emb_sum], dim=-1))
            
            # 路径 B：分两步 z(z(x, a1), a2)
            emb1 = torch.cat([torch.sin(a1), torch.cos(a1)], dim=-1)
            z_step1 = model.predictor(torch.cat([z_base, emb1], dim=-1))
            
            emb2 = torch.cat([torch.sin(a2), torch.cos(a2)], dim=-1)
            z_step2 = model.predictor(torch.cat([z_step1, emb2], dim=-1))
            
            # 计算相似度 (Cosine Similarity)
            sim = F.cosine_similarity(z_single, z_step2).mean()
            total_sim += sim.item()
            count += 1
            
    return total_sim / count if count > 0 else 0

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | Task: JEPA World Model Understanding")

    # 数据准备
    transform = transforms.ToTensor()
    mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    # 训练集：随机旋转
    train_dataset = MNIST_JEPA_Dataset(mnist_train)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    
    # 测试集：用于验证任务
    test_dataset = MNIST_JEPA_Dataset(mnist_test)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # 线性探测专用 Loader (提前创建，避免在循环中重复创建)
    train_loader_probe = DataLoader(mnist_train, batch_size=256, shuffle=True)
    test_loader_probe = DataLoader(mnist_test, batch_size=256, shuffle=False)

    # 模型初始化 - 支持 FC, Conv, ResNet 三种架构
    # model = FCJEPA(latent_dim=128).to(device)
    # model = ResNetJEPA(in_channels=1, latent_dim=128).to(device)
    model = ConvJEPA(in_channels=1, latent_dim=128).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    epochs = 10 # 增加到 10 轮，让模型有时间逃离恒等映射
    history = {
        'jepa_loss': [],
        'probe_acc': [],
        'regression_mae': [],
        'composition_sim': []
    }
    
    high_acc_mode = False # 标记是否进入高准率模式
    current_probe_acc = 0
    
    start_time = time.time()
    
    try:
        for epoch in range(1, epochs + 1):
            model.train()
            total_jepa_loss = 0
            
            for batch_idx, (img, rotated_img, _, angle_rad) in enumerate(train_loader):
                img, rotated_img, angle_rad = img.to(device), rotated_img.to(device), angle_rad.to(device)
                
                # --- JEPA 自监督训练 ---
                optimizer.zero_grad()
                z_pred, z_target, jepa_loss = model(img, rotated_img, angle_rad)
                jepa_loss.backward()
                optimizer.step()
                model.update_target(momentum=0.999) # 降低更新速度，防止目标编码器太快被同化
                
                total_jepa_loss += jepa_loss.item()
                
                # 每 200 个 Batch 进行一次采样
                if batch_idx % 200 == 0:
                    avg_loss = jepa_loss.item()
                    history['jepa_loss'].append(avg_loss)
                    
                    # 计算特征标准差，检查是否坍缩
                    with torch.no_grad():
                        z_std = z_target.std(dim=0).mean().item()
                    
                    # 动态探测逻辑：如果还没达到 88%，则每 200 batch 测一次
                    if not high_acc_mode:
                        current_probe_acc = validate_semantic_classification(model, device, train_loader_probe, test_loader_probe)
                        if current_probe_acc >= 88.0:
                            high_acc_mode = True
                            print(f"\n[System] Semantic Acc reached {current_probe_acc:.1f}%. Switching to low-frequency probing.\n")
                    
                    # 执行验证任务
                    mae = validate_action_regression(model, device, test_loader)
                    comp_sim = validate_compositionality(model, device, test_loader)
                    
                    history['probe_acc'].append(current_probe_acc)
                    history['regression_mae'].append(mae)
                    history['composition_sim'].append(comp_sim)
                    
                    print(f"Epoch {epoch} [{batch_idx}/{len(train_loader)}] Loss: {avg_loss:.4f} | "
                          f"STD: {z_std:.4f} | Acc: {current_probe_acc:.1f}% | MAE: {mae:.1f}° | Comp: {comp_sim:.3f}")

            print(f"====> Epoch: {epoch} Avg JEPA Loss: {total_jepa_loss / len(train_loader):.4f}")
              
            # 在高准率模式下，每 epoch 强制测一次
            if high_acc_mode:
                current_probe_acc = validate_semantic_classification(model, device, train_loader_probe, test_loader_probe)
                print(f"Epoch {epoch} Final Semantic Acc: {current_probe_acc:.1f}%")

        # 保存模型
        os.makedirs('outputs/models', exist_ok=True)
        torch.save(model.state_dict(), 'outputs/models/mnist_jepa.pth')
        print(f"Model saved to 'outputs/models/mnist_jepa.pth'")
            
    except KeyboardInterrupt:
        print("Training interrupted. Saving plots...")
    
    # 可视化结果
    save_plots(history)
    print(f"Training finished in {time.time() - start_time:.1f}s. Plots saved to 'outputs/results/mnist/jepa/jepa_results.png'")

def save_plots(history):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. JEPA Loss
    axes[0, 0].plot(history['jepa_loss'], color='blue', label='JEPA MSE Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_xlabel('Steps (x200)')
    axes[0, 0].grid(True)
    
    # 2. Probe Accuracy
    axes[0, 1].plot(history['probe_acc'], color='green', label='Linear Probe Acc')
    axes[0, 1].set_title('Feature Quality (Classification Acc)')
    axes[0, 1].set_ylabel('%')
    axes[0, 1].grid(True)
    
    # 3. Action Regression MAE
    axes[1, 0].plot(history['regression_mae'], color='red', label='Angle MAE')
    axes[1, 0].set_title('Action Regression (Angle Error)')
    axes[1, 0].set_ylabel('Degrees')
    axes[1, 0].grid(True)
    
    # 4. Compositionality Similarity
    axes[1, 1].plot(history['composition_sim'], color='purple', label='Cosine Similarity')
    axes[1, 1].set_title('Action Compositionality (Consistency)')
    axes[1, 1].set_ylabel('Similarity')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    os.makedirs('outputs/results/mnist/jepa', exist_ok=True)
    plt.savefig('outputs/results/mnist/jepa/jepa_results.png')
    plt.close()

if __name__ == "__main__":
    train()
