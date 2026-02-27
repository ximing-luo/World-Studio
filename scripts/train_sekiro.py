import os
import sys
import shutil
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

# 添加项目根目录到路径
path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(path)

from src.datasets.dataset import SekiroDataset
from src.model.conv_vae import ConvVAE
from src.model.spatial import SekiroVAE
from src.train.trainer import train
from src.train.untils import visualize_reconstruction, plot_loss

# 只狼 画面重建任务启动脚本 (卷积 VAE 版)
def main():
    results_dir = 'outputs/results/sekiro'
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)
    os.makedirs(results_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | Task: Sekiro Reconstruction (ConvVAE)")
    
    # 超参数
    batch_size = 96 # 卷积层显存占用较小，可以适当调大
    epochs = 30
    learning_rate = 1e-3
    beta = 1.0
    latent_dim = 196
    
    # 数据转换：仅保留基础转换，全分辨率彩色
    sekiro_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])
    
    full_dataset = SekiroDataset(data_dir='data/demos', transform=sekiro_transform)
    
    # 划分训练集和测试集
    train_size = int(0.9 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 初始化模型 (卷积 VAE)
    model = ConvVAE(latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # 添加学习率调度器：3 轮 loss 不下降则降低学习率
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=3)
    
    loss_history, bce_history, kld_history = [], [], []
    
    for epoch in range(1, epochs + 1):
        avg_loss, avg_bce, avg_kld = train(epoch, model, train_loader, optimizer, device, beta)
        loss_history.append(avg_loss)
        bce_history.append(avg_bce)
        kld_history.append(avg_kld)
        
        # 更新学习率
        # scheduler.step(avg_loss)
        
        # 可视化
        visualize_reconstruction(model, device, test_loader, epoch)
        # 将结果移动到子目录
        src_img = f'outputs/results/reconstruction_epoch_{epoch}.png'
        if os.path.exists(src_img):
            shutil.move(src_img, os.path.join(results_dir, f'reconstruction_epoch_{epoch}.png'))
    
    plot_loss(loss_history, bce_history, kld_history)
    if os.path.exists('outputs/results/loss_curve.png'):
        shutil.move('outputs/results/loss_curve.png', os.path.join(results_dir, 'loss_curve.png'))
        
    os.makedirs('outputs/models', exist_ok=True)
    torch.save(model.state_dict(), 'outputs/models/vae_sekiro_conv.pth')
    print(f"Sekiro training finished. Results in '{results_dir}'.")

if __name__ == '__main__':
    main()
