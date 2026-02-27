import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import shutil
from src.model.vae import VAE
from src.datasets.dataset import RotatedMNIST
from src.model.loss import loss_function
from src.train.untils import visualize_reconstruction, plot_loss


class Trainer:
    def __init__(self, model, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        self.model = model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.get('lr', 1e-3))
        self.beta = config.get('beta', 1.0)
        self.results_dir = config.get('results_dir', 'outputs/results')
        self.models_dir = config.get('models_dir', 'outputs/models')

        # 自动清空并创建结果目录
        self._prepare_directories()
        print(f"Using device: {self.device}")

    def _prepare_directories(self):
        """清空并创建必要的输出目录"""
        if os.path.exists(self.results_dir):
            shutil.rmtree(self.results_dir)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)

    def _get_dataloaders(self):
        """内部初始化数据加载器"""
        batch_size = self.config.get('batch_size', 128)
        angle = self.config.get('angle', 45)
        
        transform = transforms.ToTensor()
        mnist_train = datasets.MNIST('data', train=True, download=False, transform=transform)
        mnist_test = datasets.MNIST('data', train=False, download=False, transform=transform)

        train_dataset = RotatedMNIST(mnist_train, angle=angle)
        test_dataset = RotatedMNIST(mnist_test, angle=angle)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, test_loader

    def train_epoch(self, train_loader):
        self.model.train()
        train_loss = 0
        bce_loss = 0
        kld_loss = 0

        for data, target in train_loader:
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()

            recon_batch, mu, logvar = self.model(data)
            loss, BCE, KLD = loss_function(recon_batch, target, mu, logvar, self.beta)
            loss.backward()

            train_loss += loss.item()
            bce_loss += BCE.item()
            kld_loss += KLD.item()
            self.optimizer.step()

        n_samples = len(train_loader.dataset)
        return train_loss / n_samples, bce_loss / n_samples, kld_loss / n_samples

    def train(self):
        """主训练流程，使用内部配置和加载器"""
        epochs = self.config.get('epochs', 6)
        viz_interval = self.config.get('viz_interval', 2)
        save_filename = self.config.get('save_filename', 'vae_mnist.pth')

        train_loader, test_loader = self._get_dataloaders()

        loss_history, bce_history, kld_history = [], [], []

        for epoch in range(1, epochs + 1):
            avg_loss, avg_bce, avg_kld = self.train_epoch(train_loader)
            loss_history.append(avg_loss)
            bce_history.append(avg_bce)
            kld_history.append(avg_kld)

            print(f'====> Epoch: {epoch} Average loss: {avg_loss:.4f} (BCE: {avg_bce:.4f}, KLD: {avg_kld:.4f})')

            if epoch % viz_interval == 0:
                visualize_reconstruction(self.model, self.device, test_loader, epoch)

        plot_loss(loss_history, bce_history, kld_history)
        self._save_model(save_filename)
        return loss_history, bce_history, kld_history

    def _save_model(self, filename):
        """保存模型权重并打印结果路径"""
        save_path = os.path.join(self.models_dir, filename)
        torch.save(self.model.state_dict(), save_path)
        print(f"Training finished. Results saved in '{self.results_dir}/' and '{self.models_dir}/'.")
