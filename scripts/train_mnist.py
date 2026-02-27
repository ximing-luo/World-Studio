import os
import sys
import shutil
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 添加项目根目录到路径
path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(path)

from src.datasets.dataset import RotatedMNIST
from src.model.vae import VAE
from src.train.trainer import train
from src.train.untils import visualize_reconstruction, plot_loss

# MNIST 旋转预测任务启动脚本
def main():
    results_dir = 'outputs/results/mnist'
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)
    os.makedirs(results_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | Task: MNIST Rotation")
    
    # 超参数
    batch_size = 128
    epochs = 5
    learning_rate = 1e-3
    beta = 1.0
    latent_dim = 20
    hidden_dim = 400
    
    # 数据加载
    transform = transforms.ToTensor()
    mnist_train = datasets.MNIST('data', train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST('data', train=False, download=True, transform=transform)
    
    train_dataset = RotatedMNIST(mnist_train, angle=45, angle_per_digit=True)
    test_dataset = RotatedMNIST(mnist_test, angle=45, angle_per_digit=True)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 初始化模型
    model = VAE(input_dim=784, hidden_dim=hidden_dim, latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    loss_history, bce_history, kld_history = [], [], []
    
    for epoch in range(1, epochs + 1):
        avg_loss, avg_bce, avg_kld = train(epoch, model, train_loader, optimizer, device, beta)
        loss_history.append(avg_loss)
        bce_history.append(avg_bce)
        kld_history.append(avg_kld)
        
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
    torch.save(model.state_dict(), 'outputs/models/vae_mnist.pth')
    print(f"MNIST training finished. Results in '{results_dir}'.")

if __name__ == '__main__':
    main()
