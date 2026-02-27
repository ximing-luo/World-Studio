import os
import matplotlib.pyplot as plt
import torch


# 可视化预测效果
def visualize_reconstruction(model, device, test_loader, epoch):
    model.eval()
    with torch.no_grad():
        data, target = next(iter(test_loader))
        data, target = data.to(device), target.to(device)
        recon, _, _ = model(data)

        n = 8
        plt.figure(figsize=(15, 6))
        for i in range(n):
            ax = plt.subplot(3, n, i + 1)
            plt.imshow(data[i].cpu().squeeze(), cmap='gray')
            plt.title("Input (t)")
            plt.axis('off')

            ax = plt.subplot(3, n, i + 1 + n)
            plt.imshow(target[i].cpu().squeeze(), cmap='gray')
            plt.title("Target (t+1)")
            plt.axis('off')

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
