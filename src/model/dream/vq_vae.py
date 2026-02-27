import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.components.resnet import BasicBlock, ResBlock

class VectorQuantizer(nn.Module):
    """
    VQ-VAE 的核心：矢量量化层
    将连续特征映射到最近的代码簿向量
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super(VectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)

    def forward(self, inputs):
        # inputs shape: (B, C, H, W) -> (B, H, W, C)
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        flat_input = inputs.view(-1, self.embedding_dim)

        # 计算距离: (a-b)^2 = a^2 + b^2 - 2ab
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embedding.weight.t()))

        # 找到最近的索引
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # 量化
        quantized = torch.matmul(encodings, self.embedding.weight).view(inputs.shape)
        
        # 损失计算
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # 直通估计器 (Straight Through Estimator)
        quantized = inputs + (quantized - inputs).detach()
        
        return quantized.permute(0, 3, 1, 2).contiguous(), loss, encoding_indices

class VQVAE(nn.Module):
    """
    VQ-VAE 用于手写数字识别/生成任务
    """
    def __init__(self, in_channels=1, num_hiddens=64, num_residual_layers=2, 
                 num_embeddings=512, embedding_dim=32):
        super(VQVAE, self).__init__()
        
        # Encoder: 28x28 -> 14x14 -> 7x7
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, num_hiddens//2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(num_hiddens//2, num_hiddens, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(num_hiddens, embedding_dim, kernel_size=3, padding=1)
        )
        
        self.vq_layer = VectorQuantizer(num_embeddings, embedding_dim)
        
        # Decoder: 7x7 -> 14x14 -> 28x28
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embedding_dim, num_hiddens, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(num_hiddens, num_hiddens//2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(num_hiddens//2, in_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        quantized, vq_loss, _ = self.vq_layer(z)
        x_recon = self.decoder(quantized)
        return x_recon, vq_loss
