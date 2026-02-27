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

# 定义旋转数据集，模拟“世界模型”的下一帧预测
class RotatedMNIST(Dataset):
    def __init__(self, mnist_dataset, angle=45):
        self.mnist_dataset = mnist_dataset
        self.angle = angle
        # 固定旋转角度，模拟确定的物理规则
        self.rotate = transforms.RandomRotation((angle, angle))

    def __len__(self):
        return len(self.mnist_dataset)

    def __getitem__(self, idx):
        img, _ = self.mnist_dataset[idx]
        # 输入是当前帧，目标是旋转后的下一帧
        target = self.rotate(img)
        return img, target

# 定义VAE模型
class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=300, latent_dim=15):
        super(VAE, self).__init__()
        
        # 编码器：将输入压缩到隐空间
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)  # 均值 mu
        self.fc22 = nn.Linear(hidden_dim, latent_dim)  # 对数方差 logvar
        
        # 解码器：从隐空间重建输入
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)
        
    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)
    
    def reparameterize(self, mu, logvar):
        # 重参数化技巧：z = mu + eps * std
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))
    
    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# 定义损失函数：变分下界 (ELBO)
def loss_function(recon_x, x, mu, logvar, beta=1.0):
    # 1. 重建损失 (Reconstruction Loss)：衡量重建图与原图的差异
    # 使用 binary_cross_entropy，reduction='sum' 表示对所有像素求和
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    
    # 2. KL 散度 (Kullback-Leibler Divergence)：衡量隐分布与标准正态分布的差异
    # KLD = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # 总损失 = 重建损失 + beta * KL散度
    return BCE + beta * KLD, BCE, KLD

# 训练函数
def train(epoch, model, train_loader, optimizer, device, beta=1.0):
    model.train()
    train_loss = 0
    bce_loss = 0
    kld_loss = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        
        # 计算损失：重建目标变为旋转后的图
        loss, BCE, KLD = loss_function(recon_batch, target, mu, logvar, beta)
        loss.backward()
        
        train_loss += loss.item()
        bce_loss += BCE.item()
        kld_loss += KLD.item()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            # 打印每个样本的平均损失
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
                  f'Loss: {loss.item() / len(data):.4f} (BCE: {BCE.item() / len(data):.4f}, KLD: {KLD.item() / len(data):.4f})')
    
    avg_loss = train_loss / len(train_loader.dataset)
    avg_bce = bce_loss / len(train_loader.dataset)
    avg_kld = kld_loss / len(train_loader.dataset)
    
    print(f'====> Epoch: {epoch} Average loss: {avg_loss:.4f} (BCE: {avg_bce:.4f}, KLD: {avg_kld:.4f})')
    return avg_loss, avg_bce, avg_kld

# 可视化预测效果
def visualize_reconstruction(model, device, test_loader, epoch):
    model.eval()
    with torch.no_grad():
        # 获取一批测试数据：(原图, 旋转后的目标)
        data, target = next(iter(test_loader))
        data, target = data.to(device), target.to(device)
        recon, _, _ = model(data)
        
        # 取前8张图对比
        n = 8
        plt.figure(figsize=(15, 6))
        for i in range(n):
            # 1. 输入图 (原图)
            ax = plt.subplot(3, n, i + 1)
            plt.imshow(data[i].cpu().squeeze(), cmap='gray')
            plt.title("Input (t)")
            plt.axis('off')
            
            # 2. 真实目标 (旋转图)
            ax = plt.subplot(3, n, i + 1 + n)
            plt.imshow(target[i].cpu().squeeze(), cmap='gray')
            plt.title("Target (t+1)")
            plt.axis('off')
            
            # 3. 预测图 (模型输出)
            ax = plt.subplot(3, n, i + 1 + 2 * n)
            plt.imshow(recon[i].cpu().view(28, 28), cmap='gray')
            plt.title("Predict (t+1)")
            plt.axis('off')
            
        plt.tight_layout()
        os.makedirs('outputs/results', exist_ok=True)
        plt.savefig(f'outputs/results/reconstruction_epoch_{epoch}.png')
        plt.close()

# 绘制训练曲线
def plot_loss(loss_history, bce_history, kld_history):
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, label='Total Loss')
    plt.plot(bce_history, label='BCE (Reconstruction)')
    plt.plot(kld_history, label='KLD (Regularization)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('VAE Training Loss')
    plt.legend()
    plt.grid(True)
    os.makedirs('outputs/results', exist_ok=True)
    plt.savefig('outputs/results/loss_curve.png')
    plt.close()

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