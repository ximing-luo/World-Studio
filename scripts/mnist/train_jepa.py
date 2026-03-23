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

from src.world.vision.mnist import MNISTConv, MNISTResNet
from src.world.projection.projection import LinearProjection
from src.world.latents.vicreg import VicRegLatent
from src.world.predictor.predictor import MLPPredictor
from src.world.dream.jepa import TemporalPredictive
from src.datasets.mnist import MNIST_JEPA_Dataset

def validate_semantic_classification(model, device, train_loader_probe, test_loader_probe, num_epochs=1):
    """
    验证任务零：语义分类 (Semantic Classification)
    线性探测：彻底冻结特征，重新训练一个分类器，看特征的语义潜力。
    """
    model.eval()
    
    # 获取隐空间维度
    latent_dim = model.projection.token_dim
    classifier = nn.Linear(latent_dim, 10).to(device)
    
    optimizer = optim.Adam(classifier.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0
    for _ in range(num_epochs):
        classifier.train()
        for data, labels in train_loader_probe:
            data, labels = data.to(device), labels.to(device)
            with torch.no_grad():
                feat = model.vision.encode(data)
                z = model.projection.encode(feat).squeeze(1).detach()
            
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
                feat = model.vision.encode(data)
                z = model.projection.encode(feat).squeeze(1)
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
            feat_c = model.vision.encode(img)
            z_context = model.projection.encode(feat_c) # (B, 1, D)
            
            feat_t = model.target_vision.encode(rotated_img)
            z_target = model.projection.encode(feat_t) # (B, 1, D)
            
            for b in range(B):
                # 2. 对每个样本，尝试所有候选角度
                z_c = z_context[b].unsqueeze(0).expand(len(candidate_rads), -1, -1) # (N, 1, D)
                
                # 同样进行 sin/cos 编码
                cond_rads = candidate_rads.unsqueeze(1) # (N, 1)
                cond_emb = torch.cat([torch.sin(cond_rads), torch.cos(cond_rads)], dim=-1) # (N, 2)
                
                # 预测特征 (使用 TemporalPredictive 的条件注入逻辑)
                # z_c: (N, 1, D), cond_emb: (N, 2)
                # 按照 TemporalPredictive.forward 中的逻辑进行拼接
                cond = cond_emb.unsqueeze(1).expand(-1, 1, -1) # (N, 1, 2)
                z_input = torch.cat([z_c, cond], dim=-1) # (N, 1, D+2)
                z_preds = model.predictor(z_input) # (N, 1, D)
                
                # 计算与真实目标特征的距离
                distances = torch.norm(z_preds.squeeze(1) - z_target[b], dim=1)
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
            
            feat_base = model.vision.encode(img)
            z_base = model.projection.encode(feat_base) # (B, 1, D)
                    
            # 路径 A：一步到位 z(x, a1+a2)
            a_sum = (a1 + a2) % (2 * 3.14159)
            emb_sum = torch.cat([torch.sin(a_sum), torch.cos(a_sum)], dim=-1) # (B, 2)
            cond_sum = emb_sum.unsqueeze(1).expand(-1, 1, -1) # (B, 1, 2)
            z_single = model.predictor(torch.cat([z_base, cond_sum], dim=-1))
            
            # 路径 B：分两步 z(z(x, a1), a2)
            emb1 = torch.cat([torch.sin(a1), torch.cos(a1)], dim=-1)
            cond1 = emb1.unsqueeze(1).expand(-1, 1, -1)
            z_step1 = model.predictor(torch.cat([z_base, cond1], dim=-1))
            
            emb2 = torch.cat([torch.sin(a2), torch.cos(a2)], dim=-1)
            cond2 = emb2.unsqueeze(1).expand(-1, 1, -1)
            z_step2 = model.predictor(torch.cat([z_step1, cond2], dim=-1))
            
            # 计算相似度 (Cosine Similarity)
            sim = F.cosine_similarity(z_single.squeeze(1), z_step2.squeeze(1)).mean()
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
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    
    # 测试集：用于验证任务
    test_dataset = MNIST_JEPA_Dataset(mnist_test)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # 线性探测专用 Loader
    train_loader_probe = DataLoader(mnist_train, batch_size=256, shuffle=True)
    test_loader_probe = DataLoader(mnist_test, batch_size=256, shuffle=False)

    # 框架化构建模型
    latent_dim = 128
    vision = MNISTConv(in_channels=1)
    # MNISTConv 输出通道 128，H=7, W=7 (28x28 -> 7x7)
    projection = LinearProjection(in_channels=128, height=7, width=7, token_dim=latent_dim, is_vae=False)
    latent = VicRegLatent()
    # 预测器：输入潜变量 + 动作编码 (sin/cos)，输出预测潜变量
    # 注意：LinearProjection 会展平为 1x1 token，所以输入维度是 latent_dim + 2
    predictor = MLPPredictor(input_dim=latent_dim + 2, output_dim=latent_dim)
    
    model = TemporalPredictive(vision, projection, latent, predictor).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    epochs = 10
    history = {
        'jepa_loss': [],
        'probe_acc': [],
        'regression_mae': [],
        'composition_sim': []
    }
    
    high_acc_mode = False
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
                
                # 构造动作编码
                cond_emb = torch.cat([torch.sin(angle_rad), torch.cos(angle_rad)], dim=-1)
                
                # 适配框架序列输入: (B, 1, C, H, W)
                z_pred, z_target, jepa_loss = model(img.unsqueeze(1), rotated_img.unsqueeze(1), cond_emb)
                
                jepa_loss.backward()
                optimizer.step()
                model.update_target(momentum=0.999)
                
                total_jepa_loss += jepa_loss.item()
                
                if batch_idx % 200 == 0:
                    avg_loss = jepa_loss.item()
                    history['jepa_loss'].append(avg_loss)
                    
                    # 计算特征标准差，检查是否坍缩
                    with torch.no_grad():
                        z_std = z_target.std(dim=0).mean().item()
                    
                    if not high_acc_mode:
                        current_probe_acc = validate_semantic_classification(model, device, train_loader_probe, test_loader_probe)
                        if current_probe_acc >= 88.0:
                            high_acc_mode = True
                            print(f"\n[System] Semantic Acc reached {current_probe_acc:.1f}%. Switching to low-frequency probing.\n")
                    
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
