import torch
import torch.nn as nn
class StaticReconstruction(nn.Module):
    """
    静态重建框架 (Static Reconstruction Framework)。
    适用于纯 VAE 或 VQ-VAE 的训练逻辑：Image -> Feature -> Tokens -> Latent -> Image。
    """
    def __init__(self, vision, projection, latent, predictor):
        super().__init__()
        self.vision = vision
        self.projection = projection
        self.latent = latent
        self.predictor = predictor

    def forward(self, x):
        """
        x: 输入图像 (B, C, H, W)
        返回: 重建图像, 隐空间损失, 以及原始 tokens (用于 VAE 获取 mu/logvar)
        """
        # 1. 视觉编码
        feat = self.vision.encode(x)
        
        # 2. 投影成 Tokens
        tokens = self.projection.encode(feat)
        
        # 3. 隐空间约束
        z, latent_loss = self.latent(tokens)
        
        # 4. 预测器精炼 (Spatial Analysis/Refinement)
        z = self.predictor(z)
        
        # 5. 解码重构
        feat_recon = self.projection.decode(z)
        x_recon = self.vision.decode(feat_recon)
        
        return x_recon, latent_loss, tokens

    def encode(self, x):
        """仅执行编码流程"""
        feat = self.vision.encode(x)
        tokens = self.projection.encode(feat)
        z, _ = self.latent(tokens)
        z = self.predictor(z)
        return z

    def decode(self, z):
        """仅执行解码流程"""
        feat_recon = self.projection.decode(z)
        return self.vision.decode(feat_recon)
