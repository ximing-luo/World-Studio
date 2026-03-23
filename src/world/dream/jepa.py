import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
class TemporalPredictive(nn.Module):
    """
    预测性时序框架 (Predictive Temporal Framework - JEPA)。
    引入时间帧，在隐空间进行预测，但不重建像素。
    """
    def __init__(self, vision, projection, latent, predictor):
        super().__init__()
        self.vision = vision
        self.projection = projection
        self.latent = latent
        self.predictor = predictor
        
        # 目标编码器 (EMA)
        # 自动从 vision 创建副本，确保参数同步逻辑的独立性
        self.target_vision = copy.deepcopy(vision)
        # 冻结目标编码器参数
        for p in self.target_vision.parameters():
            p.requires_grad = False

    def forward(self, x_context_seq, x_target_seq, condition_seq=None):
        """
        x_context_seq: 上下文图像序列 (B, T_c, C, H, W)
        x_target_seq: 目标图像序列 (B, T_t, C, H, W)
        """
        batch_size, seq_len_c = x_context_seq.shape[:2]
        batch_size, seq_len_t = x_target_seq.shape[:2]
        num_tokens = self.projection.num_tokens
        
        # 1. 提取上下文表征
        obs_flat = x_context_seq.view(-1, *x_context_seq.shape[2:])
        h_c = self.vision.encode(obs_flat)
        z_context = self.projection.encode(h_c) # (B*T_c, S, C)
        z_context, context_penalty = self.latent(z_context)
        z_context = z_context.reshape(batch_size, seq_len_c * num_tokens, -1)
        
        # 2. 提取目标真值表征 (EMA) - 不计算梯度
        with torch.no_grad():
            target_flat = x_target_seq.view(-1, *x_target_seq.shape[2:])
            h_t = self.target_vision.encode(target_flat)
            z_target = self.projection.encode(h_t) # (B*T_t, S, C)
            z_target, target_penalty = self.latent(z_target)
            z_target = z_target.reshape(batch_size, seq_len_t * num_tokens, -1)
            
        # 3. 隐空间预测
        if condition_seq is not None:
            # 如果提供了条件，将其拼接到 tokens 维度
            # z_context: (B, T_c * num_tokens, D)
            # condition_seq: (B, T_t, D_cond) -> 需要对齐到 tokens
            B, L, D = z_context.shape
            if condition_seq.dim() == 2: # (B, D_cond)
                cond = condition_seq.unsqueeze(1).expand(-1, L, -1)
            elif condition_seq.dim() == 3: # (B, T, D_cond)
                # 简单处理：将条件扩展到每个 token
                cond = condition_seq.repeat_interleave(num_tokens, dim=1)
            else:
                cond = condition_seq
            
            z_input = torch.cat([z_context, cond], dim=-1)
            z_pred = self.predictor(z_input)
        else:
            z_pred = self.predictor(z_context)
        
        # 4. 计算预测损失 (在隐空间)
        # 这里可以使用 VicReg 风格的损失或简单的 MSE
        pred_loss = F.mse_loss(z_pred, z_target) if z_pred.shape == z_target.shape else torch.tensor(0.0)
        
        return z_pred, z_target, pred_loss + context_penalty + target_penalty

    @torch.no_grad()
    def update_target(self, momentum=0.999):
        """EMA 更新目标感知器"""
        for p_online, p_target in zip(self.vision.parameters(), self.target_vision.parameters()):
            p_target.data.mul_(momentum).add_(p_online.data, alpha=1 - momentum)
