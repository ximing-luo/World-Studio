import torch
import torch.nn as nn
import torch.nn.functional as F
class TemporalGenerative(nn.Module):
    """
    生成式时序框架 (Generative Temporal Framework - RSSM)。
    引入时间帧，支持在隐空间进行推演并重建像素未来。
    """
    def __init__(self, vision, projection, latent, predictor, action_dim=0):
        super().__init__()
        self.vision = vision
        self.projection = projection
        self.latent = latent
        self.predictor = predictor
        
        # 动作投影
        if action_dim > 0:
            # 假设动作投影到单 Token 维度
            self.action_proj = nn.Linear(action_dim, projection.token_dim)

    def observe(self, obs_seq, action_seq=None):
        """
        处理已知观测序列。
        obs_seq: (B, T, C, H, W)
        返回: 隐状态序列, 后验分布采样 z, 隐空间损失
        """
        batch_size, seq_len = obs_seq.shape[:2]
        num_tokens = self.projection.num_tokens
        
        # 1. 编码所有观测
        obs_flat = obs_seq.view(-1, *obs_seq.shape[2:])
        feat = self.vision.encode(obs_flat)
        tokens = self.projection.encode(feat) # (B*T, S, C)
        
        # 2. 隐空间约束 (获取后验)
        z, latent_loss = self.latent(tokens) # z: (B*T, S, C)
        z = z.reshape(batch_size, seq_len * num_tokens, -1)
        
        # 3. 隐空间推演
        # 注意：这里 predictor 现在直接返回预测值，我们假设输出即为先验参数
        prior_params = self.predictor(z)
            
        return prior_params, z, latent_loss

    def imagine_next(self, prev_states_tokens, action=None):
        """
        想象下一帧。
        prev_states_tokens: 历史潜变量序列 (B, T*S, C)
        """
        # 预测先验分布参数
        prior_params = self.predictor(prev_states_tokens)
            
        # 取最后一步对应的空间 tokens 的参数
        num_tokens = self.projection.num_tokens
        last_prior_params = prior_params[:, -num_tokens:, :]
        
        # 使用 latent 组件进行采样
        mu, logvar = torch.chunk(last_prior_params, 2, dim=-1)
        s_next = self.latent.sample(mu, logvar) if hasattr(self.latent, 'sample') else mu
        
        return s_next, (mu, logvar)

    def decode(self, z):
        """从潜变量重构图像"""
        # z: (B, S, C)
        feat_map = self.projection.decode(z)
        return self.vision.decode(feat_map)
