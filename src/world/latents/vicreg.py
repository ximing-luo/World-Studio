import torch
import torch.nn as nn
import torch.nn.functional as F

class VicRegLatent(nn.Module):
    """
    确定性隐空间约束层。
    使用 VicReg (Variance-Invariance-Covariance Regularization) 防止坍缩。
    """
    def __init__(self, var_weight=1.0, cov_weight=0.01):
        super().__init__()
        self.var_weight = var_weight
        self.cov_weight = cov_weight

    def forward(self, tokens):
        """
        tokens: (B, S, C)
        返回: 原始 tokens, 以及 VicReg 统计损失。
        """
        # 计算方差损失 (Variance Loss): 维持表征分布
        # tokens: (B, S, C) -> 在 Batch 维度计算方差
        # 为了计算方便，我们将 B 和 S 合并，或者只在 B 维度计算
        flat_tokens = tokens.reshape(-1, tokens.size(-1))
        
        std = torch.sqrt(flat_tokens.var(dim=0) + 1e-04)
        var_loss = torch.mean(F.relu(1.0 - std))
        
        # 计算协方差损失 (Covariance Loss): 去冗余
        batch_size = flat_tokens.size(0)
        dim = flat_tokens.size(1)
        
        # 中心化
        z = flat_tokens - flat_tokens.mean(dim=0)
        cov = (z.T @ z) / (batch_size - 1)
        
        # 取非对角线元素
        def off_diagonal(x):
            n, m = x.shape
            assert n == m
            return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

        cov_loss = off_diagonal(cov).pow_(2).sum() / dim
        
        total_penalty = self.var_weight * var_loss + self.cov_weight * cov_loss
        
        return tokens, total_penalty
