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
        
        # 自动推断图像形状
        # 如果是 MNIST [1, 28, 28], 展平后 784
        # 如果是 Sekiro [3, 136, 240], 展平后 97920
        img_shape = data[0].shape
        
        n = min(8, data.size(0))
        plt.figure(figsize=(15, 6))
        for i in range(n):
            ax = plt.subplot(3, n, i + 1)
            img_input = data[i].cpu().squeeze()
            if img_input.dim() == 3: # (C, H, W) -> (H, W, C) for imshow
                img_input = img_input.permute(1, 2, 0)
                # 如果是 float tensor (0-1)，直接显示；如果是 uint8 (0-255)，转换为 0-1
                if img_input.dtype == torch.float32:
                    img_input = torch.clamp(img_input, 0, 1)
            plt.imshow(img_input.numpy(), cmap='gray' if img_input.dim() == 2 else None)
            plt.title("Input")
            plt.axis('off')

            ax = plt.subplot(3, n, i + 1 + n)
            img_target = target[i].cpu().squeeze()
            if img_target.dim() == 3:
                img_target = img_target.permute(1, 2, 0)
                if img_target.dtype == torch.float32:
                    img_target = torch.clamp(img_target, 0, 1)
            plt.imshow(img_target.numpy(), cmap='gray' if img_target.dim() == 2 else None)
            plt.title("Target")
            plt.axis('off')

            ax = plt.subplot(3, n, i + 1 + 2 * n)
            img_recon = recon[i].cpu().view(img_shape).squeeze()
            if img_recon.dim() == 3:
                img_recon = img_recon.permute(1, 2, 0)
                if img_recon.dtype == torch.float32:
                    img_recon = torch.clamp(img_recon, 0, 1)
            plt.imshow(img_recon.numpy(), cmap='gray' if img_recon.dim() == 2 else None)
            plt.title("Predict")
            plt.axis('off')

        plt.tight_layout()
        os.makedirs('outputs/results', exist_ok=True)
        plt.savefig(f'outputs/results/reconstruction_epoch_{epoch}.png')
        plt.close()


# 绘制训练曲线
def plot_loss(loss_history, bce_history, kld_history):
    plt.figure(figsize=(10, 5))
    # 只画归一化后的指标，因为 Total Loss 量级太大，画在一起会看不清
    plt.plot(bce_history, label='BCE/px (Reconstruction)')
    plt.plot(kld_history, label='KLD/dim (Regularization)')
    plt.xlabel('Epoch')
    plt.ylabel('Normalized Loss')
    plt.title('VAE Training Loss (Normalized Metrics)')
    plt.legend()
    plt.grid(True)
    os.makedirs('outputs/results', exist_ok=True)
    plt.savefig('outputs/results/loss_curve.png')
    plt.close()
