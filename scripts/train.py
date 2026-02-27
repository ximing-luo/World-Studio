import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil

# 主程序
def main():
    # 自动清空旧的日志和结果
    results_dir = 'outputs/results'
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)
    os.makedirs(results_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 超参数
    batch_size = 128
    epochs = 6
    learning_rate = 1e-3
    beta = 1.0  # KL损失的权重
    
    # 数据加载：包装成旋转预测数据集
    transform = transforms.ToTensor()
    mnist_train = datasets.MNIST('data', train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST('data', train=False, download=True, transform=transform)
    
    # 旋转角度设为 45 度
    train_dataset = RotatedMNIST(mnist_train, angle=45)
    test_dataset = RotatedMNIST(mnist_test, angle=45)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 初始化模型和优化器
    model = VAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    loss_history = []
    bce_history = []
    kld_history = []
    
    # 训练
    for epoch in range(1, epochs + 1):
        avg_loss, avg_bce, avg_kld = train(epoch, model, train_loader, optimizer, device, beta)
        loss_history.append(avg_loss)
        bce_history.append(avg_bce)
        kld_history.append(avg_kld)
        
        # 每隔 2 个 epoch 可视化一次重建效果
        if epoch % 2 == 0:
            visualize_reconstruction(model, device, test_loader, epoch)
    
    # 绘制最终损失曲线
    plot_loss(loss_history, bce_history, kld_history)
    
    # 保存模型
    os.makedirs('outputs/models', exist_ok=True)
    torch.save(model.state_dict(), 'outputs/models/vae_mnist.pth')
    print("Training finished. Results saved in 'outputs/results/' and 'outputs/models/'.")

if __name__ == '__main__':
    main()