import torch
import torch.nn as nn
import torch.nn.functional as F

class VQLatent(nn.Module):
    """
    VQ (Vector Quantizer) 隐空间约束层。
    负责离散量化以及量化损失计算。
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.num_embeddings, 1.0 / self.num_embeddings)

    def forward(self, tokens):
        """
        tokens: (B, S, C) - 投影后的连续 Token。
        返回: 量化后的 tokens, 以及 VQ 损失。
        """
        # inputs: (B, S, C)
        flat_input = tokens.reshape(-1, self.embedding_dim)
        
        # 计算 L2 距离
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embedding.weight.t()))

        # 找到最近的索引
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=tokens.device)
        encodings.scatter_(1, encoding_indices, 1)

        # 量化
        quantized = torch.matmul(encodings, self.embedding.weight).view(tokens.shape)

        # 损失计算
        e_latent_loss = F.mse_loss(quantized.detach(), tokens)
        q_latent_loss = F.mse_loss(quantized, tokens.detach())
        vq_loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # 直通估计 (Straight Through Estimator)
        quantized = tokens + (quantized - tokens).detach()
        
        return quantized, vq_loss

    def get_indices(self, tokens):
        """获取最近邻索引"""
        flat_input = tokens.reshape(-1, self.embedding_dim)
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embedding.weight.t()))
        return torch.argmin(distances, dim=1).view(tokens.shape[:2])
