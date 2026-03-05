import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseJEPA(nn.Module):
    """
    JEPA 基类。
    vision: 编码器。
    projection: 投影层。
    target_vision: 目标编码器 (EMA)。
    """
    def __init__(self, vision, projection, target_vision, predictor):
        super(BaseJEPA, self).__init__()
        self.vision = vision
        self.projection = projection
        self.target_vision = target_vision
        self.predictor = predictor
        self.latent_dim = projection.latent_dim

    def encode_context(self, x):
        """上下文编码 -> Tokens"""
        h = self.vision.encode(x)
        return self.projection.encode(h)

    def encode_target(self, x):
        """目标编码 -> Tokens"""
        h = self.target_vision.encode(x)
        return self.projection.encode(h)

    def forward(self, x_context, x_target, condition):
        # 1. 提取上下文表征 (B, S, C)
        z_context = self.encode_context(x_context)
        
        # 2. 提取目标表征 - 不计算梯度
        with torch.no_grad():
            z_target = self.encode_target(x_target)
            
        # 3. 处理条件输入 (这里假设 condition 作用于所有 tokens 或作为额外 token)
        # 简化处理：将 condition 拼接到每个 token
        if len(condition.shape) == 1:
            angle_emb = torch.stack([torch.sin(condition), torch.cos(condition)], dim=-1)
        else:
            angle_emb = condition
        
        # 扩展 angle_emb 匹配 tokens 数量
        angle_emb = angle_emb.unsqueeze(1).expand(-1, z_context.size(1), -1)
            
        # 4. 预测目标表征
        z_target_pred = self.predictor(torch.cat([z_context, angle_emb], dim=-1))
        
        # 5. 计算损失 (Flatten 整个序列计算)
        total_loss = self.calculate_loss(
            z_context.view(z_context.size(0), -1), 
            z_target.view(z_target.size(0), -1), 
            z_target_pred.view(z_target_pred.size(0), -1)
        )
        
        return z_target_pred, z_target, total_loss

    def calculate_loss(self, z_context, z_target, z_target_pred):
        """
        VicReg Loss 计算。
        z_context, z_target, z_target_pred: 表征向量 (已拉平)
        """
        # (1) Invariance Loss: 预测与真实的均方误差
        sim_loss = F.mse_loss(z_target_pred, z_target)
        
        # (2) Variance Loss: 防止崩溃，维持表征分布
        std_context = torch.sqrt(z_context.var(dim=0) + 1e-04)
        std_target = torch.sqrt(z_target.var(dim=0) + 1e-04)
        std_loss = torch.mean(F.relu(1.0 - std_context)) + torch.mean(F.relu(1.0 - std_target))
        
        # (3) Covariance Loss: 去冗余，使表征维度独立
        def off_diagonal(x):
            n, m = x.shape
            assert n == m
            return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

        c = (z_context.T @ z_context) / (z_context.shape[0] - 1)
        cov_loss = off_diagonal(c).pow_(2).sum() / z_context.shape[1]

        return sim_loss + 1.0 * std_loss + 0.01 * cov_loss

    @torch.no_grad()
    def update_target(self, momentum=0.999):
        """EMA 更新目标感知器的编码器部分"""
        for p_context, p_target in zip(self.vision.parameters(), self.target_vision.parameters()):
            p_target.data.mul_(momentum).add_(p_context.data, alpha=1 - momentum)

class AttentiveJEPA(BaseJEPA):
    """
    [演进版] 注意力驱动的 JEPA。
    """
    def __init__(self, vision, projection, target_vision, config):
        super(AttentiveJEPA, self).__init__(vision, projection, target_vision, None)
        
        from src.model.backbone.transform import DeepSeekV3Block
        
        # 升级 Predictor 为 Transformer Blocks
        self.transformer_predictor = nn.ModuleList([
            DeepSeekV3Block(config) for _ in range(config.num_layers)
        ])
        
        # 潜空间 Token 维度
        token_dim = projection.latent_dim // projection.num_tokens
        
        # 输入投影：Token -> Transformer Hidden Dim
        self.input_proj = nn.Linear(token_dim, config.hidden_dim) if token_dim != config.hidden_dim else nn.Identity()
        
        # 最后的投影头：Transformer 输出 -> 目标 Token 维度
        self.predictor_head = nn.Linear(config.hidden_dim, token_dim)

    def forward(self, x_context_seq, x_target_seq, condition_seq=None):
        """
        处理序列化的上下文预测。
        x_context_seq: (Batch, Seq, Channels, H, W)
        """
        batch_size, seq_len = x_context_seq.shape[:2]
        
        # 1. 提取上下文表征 (B, Seq, Tokens_Per_Frame, Dim)
        obs_flat = x_context_seq.view(-1, *x_context_seq.shape[2:])
        h = self.vision.encode(obs_flat)
        z_context = self.projection.encode(h) # (B*Seq, S, C)
        z_context = z_context.reshape(batch_size, seq_len * self.projection.num_tokens, -1)
        
        # 2. 提取目标真值表征 (EMA)
        with torch.no_grad():
            target_flat = x_target_seq.view(-1, *x_target_seq.shape[2:])
            h_target = self.target_vision.encode(target_flat)
            z_target = self.projection.encode(h_target) # (B*Seq, S, C)
            z_target = z_target.reshape(batch_size, seq_len * self.projection.num_tokens, -1)
            
        # 3. Transformer 潜空间推演
        h = self.input_proj(z_context)
        for block in self.transformer_predictor:
            h = block(h)
            
        # 4. 投影到目标空间
        z_target_pred = self.predictor_head(h)
        
        # 5. 计算损失
        loss = self.calculate_loss(
            z_context.reshape(batch_size, -1), 
            z_target.reshape(batch_size, -1), 
            z_target_pred.reshape(batch_size, -1)
        )
        
        return z_target_pred, z_target, loss
