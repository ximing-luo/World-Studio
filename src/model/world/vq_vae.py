import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    """
    VQ-VAE 核心量化层。
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super(VectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.num_embeddings, 1.0 / self.num_embeddings)

    def forward(self, inputs, is_spatial=False):
        # inputs: (B, S, C)
        
        # 计算 L2 距离
        # (B*S, C)
        flat_input = inputs.view(-1, self.embedding_dim)
        
        # distances = (a - b)^2 = a^2 + b^2 - 2ab
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

        # 直通估计 (Straight Through Estimator)
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantized, loss, perplexity

class BaseVQVAE(nn.Module):
    """
    VQ-VAE 基类。
    vision: 提供 encode/decode 特征图的能力。
    projection: 提供特征图与潜空间 Token 之间的双向映射。
    """
    def __init__(self, vision, projection, num_embeddings=512, commitment_cost=0.25):
        super(BaseVQVAE, self).__init__()
        self.vision = vision
        self.projection = projection
        # token_dim = latent_dim // num_tokens
        token_dim = projection.latent_dim // projection.num_tokens
        self.vq_layer = VectorQuantizer(num_embeddings, token_dim, commitment_cost)

    def encode(self, x):
        h = self.vision.encode(x)
        z = self.projection.encode(h)
        return z

    def decode(self, z):
        h = self.projection.decode(z)
        return self.vision.decode(h)

    def forward(self, x):
        z = self.encode(x)
        quantized, vq_loss, _ = self.vq_layer(z, is_spatial=False)
        x_recon = self.decode(quantized)
        return x_recon, vq_loss
