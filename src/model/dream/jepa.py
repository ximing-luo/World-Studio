import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.components.resnet import BasicBlock

class MNIST_JEPA(nn.Module):
    """
    Joint-Embedding Predictive Architecture (JEPA) 简化版
    用于研究 MNIST 的特征预测任务（例如预测旋转后的特征）
    """
    def __init__(self, in_channels=1, latent_dim=128):
        super(MNIST_JEPA, self).__init__()
        
        # Context Encoder (x)
        self.context_encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1), # 14x14
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # 7x7
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, latent_dim)
        )
        
        # Target Encoder (y) - 在实际训练中通常使用 EMA 更新
        self.target_encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, latent_dim)
        )
        
        # Predictor: z_context + condition -> z_target_pred
        # 使用 sin/cos 编码 condition (2维) 以增强角度信号
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim + 2, latent_dim * 2),
            nn.LayerNorm(latent_dim * 2),
            nn.SiLU(),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim)
        )

    def forward(self, x_context, x_target, condition):
        # condition: (B, 1) 弧度
        # 1. 提取上下文表征
        z_context = self.context_encoder(x_context)
        
        # 2. 提取目标表征 (用于计算损失)
        with torch.no_grad():
            z_target = self.target_encoder(x_target)
            
        # 3. 增强角度信号：将弧度转换为 [sin, cos]
        angle_emb = torch.cat([torch.sin(condition), torch.cos(condition)], dim=-1)
            
        # 4. 预测目标表征
        z_target_pred = self.predictor(torch.cat([z_context, angle_emb], dim=-1))
        
        # --- 复合损失计算 ---
        
        # (1) Invariance Loss: 预测值要接近目标值
        sim_loss = F.mse_loss(z_target_pred, z_target)
        
        # (2) Variance Loss: 防止坍缩。强制 Batch 内特征的标准差维持在 1.0 附近
        # 计算 z_context 和 z_target 的标准差
        std_context = torch.sqrt(z_context.var(dim=0) + 1e-04)
        std_target = torch.sqrt(z_target.var(dim=0) + 1e-04)
        std_loss = torch.mean(F.relu(1.0 - std_context)) + torch.mean(F.relu(1.0 - std_target))
        
        # (3) Covariance Loss (可选): 减少冗余，让特征每一维尽量独立
        def off_diagonal(x):
            n, m = x.shape
            assert n == m
            return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

        c = (z_context.T @ z_context) / (z_context.shape[0] - 1)
        cov_loss = off_diagonal(c).pow_(2).sum() / z_context.shape[1]

        # 总损失：平衡预测精度和表征丰富度
        total_loss = sim_loss + 1.0 * std_loss + 0.01 * cov_loss
        
        return z_target_pred, z_target, total_loss

    @torch.no_grad()
    def update_target(self, momentum=0.999):
        """
        使用 EMA (Exponential Moving Average) 更新目标编码器
        target = momentum * target + (1 - momentum) * context
        """
        for p_context, p_target in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
            p_target.data.mul_(momentum).add_(p_context.data, alpha=1 - momentum)
