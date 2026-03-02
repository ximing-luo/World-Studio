import os
import sys
import time
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

# 添加项目根目录到路径
path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(path)

from src.datasets.sekiro import Sekiro_VQVAE_Dataset
from src.model.sekiro.vq_vae import ConvSekiroVQVAE, ResNetSekiroVQVAE
from src.train.train_utils import get_log_dir

def train(epoch, model, train_loader, optimizer, device, embedding_dim=64, writer=None):
    model.train()
    total_loss = 0
    start_time = time.time()
    
    for batch_idx, (data, target, action) in enumerate(train_loader):
        # 将 uint8 数据搬运到 GPU 后再进行浮点转换和归一化 (极致性能)
        data = data.to(device).float() / 255.0
        target = target.to(device).float() / 255.0
        B, C, H, W = data.size()
        
        optimizer.zero_grad()
        
        # 任务：输入当前帧，预测/重构下一帧
        recon_batch, vq_loss = model(data)
        
        # 重建损失 (对比下一帧)
        recon_loss = F.mse_loss(recon_batch, target)
        loss = recon_loss + vq_loss
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 10 == 0:
            # 计算用于评估的平均值
            # 1. Pixel MSE: VQ-VAE 的 F.mse_loss 默认就是 mean 模式，所以直接使用
            pixel_mse = recon_loss.item()
            # 2. VQ per Dim: 将 VQ 损失平摊到隐维度
            vq_per_dim = vq_loss.item() / embedding_dim
            
            elapsed = time.time() - start_time
            print(f'[{elapsed:.2f}s] Train Epoch: {epoch} [{batch_idx * B}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
                  f'Loss: {loss.item():.4f} (Pixel MSE: {pixel_mse:.6f}, VQ/Dim: {vq_per_dim:.6f})')
            
            if writer is not None:
                step = (epoch - 1) * len(train_loader) + batch_idx
                writer.add_scalar('train/loss', loss.item(), step)
                writer.add_scalar('train/recon_loss', recon_loss.item(), step)
                writer.add_scalar('train/vq_loss', vq_loss.item(), step)
    
    avg_loss = total_loss / len(train_loader)
    print(f'====> Epoch: {epoch} Average loss: {avg_loss:.4f}')
    return avg_loss

def visualize(model, loader, device, epoch, writer=None):
    model.eval()
    with torch.no_grad():
        data, target, action = next(iter(loader))
        data = data.to(device).float() / 255.0
        target = target.to(device).float() / 255.0
        recon, _ = model(data)
        
        n = min(data.size(0), 5)
        fig, axes = plt.subplots(3, n, figsize=(n*3, 9))
        
        for i in range(n):
            # Input
            axes[0, i].imshow(data[i].cpu().permute(1, 2, 0).numpy())
            axes[0, i].axis('off')
            if i == 0: axes[0, i].set_title('Original (t)')
            
            # Target
            axes[1, i].imshow(target[i].cpu().permute(1, 2, 0).numpy())
            axes[1, i].axis('off')
            if i == 0: axes[1, i].set_title('Target (t+10)')
            
            # Reconstruction
            axes[2, i].imshow(recon[i].cpu().permute(1, 2, 0).numpy())
            axes[2, i].axis('off')
            if i == 0: axes[2, i].set_title('VQ-Predicted (t+10)')
            
        plt.tight_layout()
        os.makedirs('outputs/results/sekiro/vqvae', exist_ok=True)
        plt.savefig(f'outputs/results/sekiro/vqvae/epoch_{epoch}.png')
        plt.close()
        print(f"Saved visualization to outputs/results/sekiro/vqvae/epoch_{epoch}.png")

def eval_task(model, loader, device, writer=None, epoch=None):
    """
    验证任务：计算验证集上的重建损失和 VQ 损失
    """
    model.eval()
    total_recon_loss = 0
    total_vq_loss = 0
    total_loss = 0
    
    with torch.no_grad():
        for data, target, action in loader:
            data = data.to(device).float() / 255.0
            target = target.to(device).float() / 255.0
            
            recon_batch, vq_loss = model(data)
            recon_loss = F.mse_loss(recon_batch, target)
            loss = recon_loss + vq_loss
            
            total_recon_loss += recon_loss.item()
            total_vq_loss += vq_loss.item()
            total_loss += loss.item()
            
    avg_loss = total_loss / len(loader)
    avg_recon = total_recon_loss / len(loader)
    avg_vq = total_vq_loss / len(loader)
    
    print(f'====> Validation Loss: {avg_loss:.4f} (Recon: {avg_recon:.4f}, VQ: {avg_vq:.4f})')
    
    if writer is not None and epoch is not None:
        writer.add_scalar('val/loss', avg_loss, epoch)
        writer.add_scalar('val/recon_loss', avg_recon, epoch)
        writer.add_scalar('val/vq_loss', avg_vq, epoch)
        
    return avg_loss

def main():
    # Hyperparameters
    batch_size = 64
    learning_rate = 1e-3
    epochs = 10
    prediction_offset = 1 # 预测未来第几帧
    num_hiddens = 128
    num_embeddings = 512
    embedding_dim = 64
    log_dir = get_log_dir('logs/sekiro/vqvae')
    
    # DataLoader num_workers: Windows 下开启多进程会导致内存占用飙升 (每个 worker 复制一份库文件内存)
    # 如果内存不足 (看到 7GB+)，建议设为 0 或 1；若内存充足想要加速，可设为 2-4
    num_workers = 1 

    results_dir = 'outputs/results/sekiro/vqvae'
    os.makedirs(results_dir, exist_ok=True)
    
    # TensorBoard Writer
    writer = SummaryWriter(log_dir)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | Task: Sekiro VQ-VAE Next-Frame Prediction")
    
    # 数据准备
    dataset = Sekiro_VQVAE_Dataset(frame_skip=prediction_offset, data_dir='data/Sekiro/recordings')
    # 简单划分训练集和验证集 (使用相同的数据集对象，但在 loader 中 shuffle 不同)
    # 在实际项目中应严格划分，这里为了演示方便复用 dataset
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    # 模型初始化 - 支持 Conv, ResNet 架构
    # model = ResNetSekiroVQVAE(in_channels=3, num_hiddens=num_hiddens, num_embeddings=num_embeddings, embedding_dim=embedding_dim).to(device)
    model = ConvSekiroVQVAE(in_channels=3, num_hiddens=num_hiddens, num_embeddings=num_embeddings, embedding_dim=embedding_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(1, epochs + 1):
        train(epoch, model, train_loader, optimizer, device, embedding_dim, writer)
        eval_task(model, test_loader, device, writer, epoch)
        visualize(model, train_loader, device, epoch, writer)

    # 保存模型
    os.makedirs('outputs/models', exist_ok=True)
    torch.save(model.state_dict(), 'outputs/models/sekiro_vqvae.pth')
    print(f"Model saved to 'outputs/models/sekiro_vqvae.pth'")
    
    writer.close()

if __name__ == '__main__':
    main()
