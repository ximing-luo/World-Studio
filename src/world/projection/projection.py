import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseProjection(nn.Module):
    """
    投影头基类：定义潜空间与特征图之间的双向映射。
    所有子类 encode 必须返回 (B, Tokens, Dim)。
    is_vae: 如果为 True，encode 输出维度翻倍（用于 VAE 的 mu 和 logvar）。
    """
    def __init__(self, in_channels, token_dim, num_tokens=1, is_spatial=False, is_vae=False):
        super().__init__()
        self.in_channels = in_channels
        self.token_dim = token_dim # 单个 Token 的维度
        self.num_tokens = num_tokens # Token 数量
        self.latent_dim = token_dim * num_tokens # 总潜空间维度
        
        self.is_spatial = is_spatial
        self.is_vae = is_vae
        
        # 实际输出维度：如果是 VAE 则单 Token 维度翻倍（用于 mu 和 logvar）
        self.out_token_dim = token_dim * 2 if is_vae else token_dim

    def encode(self, x):
        """编码：Feature Map -> Tokens"""
        raise NotImplementedError

    def decode(self, tokens):
        """解码：Tokens -> Feature Map"""
        raise NotImplementedError

class LinearProjection(BaseProjection):
    """方案 A: 向量潜空间 (Vector Latent) - 全局拉平"""
    def __init__(self, in_channels, height, width, token_dim, is_vae=False):
        super().__init__(in_channels, token_dim, num_tokens=1, is_spatial=False, is_vae=is_vae)
        self.h, self.w = height, width
        self.proj = nn.Linear(in_channels * height * width, self.out_token_dim)
        self.inv_proj = nn.Linear(token_dim, in_channels * height * width)

    def encode(self, x):
        # x: (B, C, H, W) -> (B, 1, Out_Token_Dim)
        b = x.size(0)
        h = self.proj(x.flatten(1))
        return h.unsqueeze(1)

    def decode(self, tokens):
        # tokens: (B, 1, Token_Dim) -> (B, C, H, W)
        b = tokens.size(0)
        h = self.inv_proj(tokens.squeeze(1))
        return h.view(b, self.in_channels, self.h, self.w)

class AttentionProjection(BaseProjection):
    """方案 B: 注意力池化 - 提取全局抽象语义"""
    def __init__(self, in_channels, height, width, token_dim, num_heads=8, is_vae=False):
        super().__init__(in_channels, token_dim, num_tokens=1, is_spatial=False, is_vae=is_vae)
        from src.model.components.attention import AttentionPooling
        self.h, self.w = height, width
        self.pool = AttentionPooling(in_channels, num_heads)
        self.proj = nn.Linear(in_channels, self.out_token_dim)
        self.inv_proj = nn.Linear(token_dim, in_channels * height * width)

    def encode(self, x):
        # x: (B, C, H, W) -> (B, 1, Out_Token_Dim)
        h = self.pool(x) # (B, C)
        return self.proj(h).unsqueeze(1)

    def decode(self, tokens):
        # tokens: (B, 1, Token_Dim) -> (B, C, H, W)
        b = tokens.size(0)
        h = self.inv_proj(tokens.squeeze(1))
        return h.view(b, self.in_channels, self.h, self.w)

class SpatialProjection(BaseProjection):
    """方案 C: 空间潜空间 (Spatial Latent) - 保持二维物理结构"""
    def __init__(self, in_channels, token_dim, height=4, width=8, is_vae=False):
        super().__init__(in_channels, token_dim, num_tokens=height * width, is_spatial=True, is_vae=is_vae)
        self.h, self.w = height, width
        
        # 1x1 卷积压缩通道，保持空间结构
        self.proj = nn.Conv2d(in_channels, self.out_token_dim, 1)
        self.inv_proj = nn.Conv2d(token_dim, in_channels, 1)

    def encode(self, x):
        # x: (B, C, H, W) -> (B, H*W, Token_Dim[*2])
        b = x.size(0)
        h = self.proj(x) # (B, Token_Dim[*2], H, W)
        return h.flatten(2).permute(0, 2, 1)

    def decode(self, tokens):
        # tokens: (B, H*W, Token_Dim) -> (B, C, H, W)
        b = tokens.size(0)
        h = tokens.permute(0, 2, 1).view(b, self.token_dim, self.h, self.w)
        return self.inv_proj(h)
