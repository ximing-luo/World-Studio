import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseRSSM(nn.Module):
    """
    RSSM 基类。
    vision: 提供特征提取能力。
    projection: 提供特征图与潜空间 Token 的双向映射。
    """
    def __init__(self, vision, projection, deterministic_dim=256, stochastic_dim=32, action_dim=0):
        super(BaseRSSM, self).__init__()
        self.vision = vision
        self.projection = projection
        self.det_dim = deterministic_dim
        self.stoch_dim = stochastic_dim # 这里的 stoch_dim 对应总潜变量维度
        self.act_dim = action_dim

        # 1. 核心确定性路径：GRU (处理拉平后的全局状态)
        self.rnn_cell = nn.GRUCell(stochastic_dim + action_dim, deterministic_dim)

        # 2. 先验预测
        self.prior_net = nn.Sequential(
            nn.Linear(deterministic_dim, deterministic_dim),
            nn.SiLU(),
            nn.Linear(deterministic_dim, stochastic_dim * 2)
        )

        # 3. 后验纠偏 (延迟初始化)
        self.post_net = None 

    def init_post_net(self, obs_feat_dim):
        self.post_net = nn.Sequential(
            nn.Linear(self.det_dim + obs_feat_dim, self.det_dim),
            nn.SiLU(),
            nn.Linear(self.det_dim, self.stoch_dim * 2)
        )
        return self.post_net

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def observe(self, obs, prev_state=None, action=None):
        batch_size = obs.size(0)
        if prev_state is None:
            prev_state = self.init_state(batch_size, obs.device)
        
        h_prev, s_prev = prev_state
        
        # RNN 状态更新
        rnn_input = s_prev
        if action is not None:
            rnn_input = torch.cat([s_prev, action], dim=-1)
        h_t = self.rnn_cell(rnn_input, h_prev)

        # 提取观测特征并投影为 Token (B, S, C)
        obs_feat_map = self.vision.encode(obs)
        obs_tokens = self.projection.encode(obs_feat_map)
        obs_feat = obs_tokens.view(batch_size, -1) # 拉平用于基础 RSSM 处理

        if self.post_net is None:
            self.init_post_net(obs_feat.size(-1)).to(obs.device)

        # 后验分布采样
        post_params = self.post_net(torch.cat([h_t, obs_feat], dim=-1))
        mu, logvar = torch.chunk(post_params, 2, dim=-1)
        s_t = self.reparameterize(mu, logvar)

        return (h_t, s_t), (mu, logvar)

    def imagine(self, prev_state, action=None):
        h_prev, s_prev = prev_state
        
        rnn_input = s_prev
        if action is not None:
            rnn_input = torch.cat([s_prev, action], dim=-1)
        h_t = self.rnn_cell(rnn_input, h_prev)

        # 先验预测
        prior_params = self.prior_net(h_t)
        mu, logvar = torch.chunk(prior_params, 2, dim=-1)
        s_t = self.reparameterize(mu, logvar)

        return (h_t, s_t), (mu, logvar)

    def decode_state(self, h, s):
        """从世界状态重构观测"""
        # s: (B, Stoch_Dim) -> (B, S, C)
        z = s.view(s.size(0), self.projection.num_tokens, -1)
        feat_map = self.projection.decode(z)
        return self.vision.decode(feat_map)

    def init_state(self, batch_size, device):
        h = torch.zeros(batch_size, self.det_dim, device=device)
        s = torch.zeros(batch_size, self.stoch_dim, device=device)
        return h, s

class AttentiveRSSM(BaseRSSM):
    """
    [演进版] 注意力驱动的状态空间模型。
    """
    def __init__(self, vision, projection, config, deterministic_dim=256, stochastic_dim=32, action_dim=0):
        super(AttentiveRSSM, self).__init__(vision, projection, deterministic_dim, stochastic_dim, action_dim)
        
        from src.model.backbone.transform import DeepSeekV3Block
        self.rnn_cell = None # 禁用 GRU
        
        self.dynamics_transformer = nn.ModuleList([
            DeepSeekV3Block(config) for _ in range(config.num_layers)
        ])
        
        # 潜空间 Token 维度
        token_dim = stochastic_dim // projection.num_tokens
        
        # 输入投影：Token -> Transformer Hidden Dim
        self.input_proj = nn.Linear(token_dim, config.hidden_dim) if token_dim != config.hidden_dim else nn.Identity()
        
        if action_dim > 0:
            self.action_proj = nn.Linear(action_dim, token_dim)
        
        # 最后的表征头：Transformer 输出 -> 先验分布参数
        self.prior_head = nn.Sequential(
            nn.Linear(config.hidden_dim, deterministic_dim),
            nn.SiLU(),
            nn.Linear(deterministic_dim, token_dim * 2)
        )

    def init_post_net(self, obs_feat_dim):
        # 对于 AttentiveRSSM，post_net 处理单个 token
        token_dim = self.stoch_dim // self.projection.num_tokens
        self.post_net = nn.Sequential(
            nn.Linear(self.det_dim + obs_feat_dim, self.det_dim),
            nn.SiLU(),
            nn.Linear(self.det_dim, token_dim * 2)
        )
        return self.post_net

    def observe(self, obs_seq, action_seq=None):
        """处理序列输入"""
        batch_size, seq_len = obs_seq.shape[:2]
        num_tokens = self.projection.num_tokens
        
        # 1. 视觉编码 + 投影: (B*T, C, H, W) -> (B*T, S, C)
        obs_flat = obs_seq.view(-1, *obs_seq.shape[2:])
        h_feat = self.vision.encode(obs_flat)
        obs_tokens = self.projection.encode(h_feat) # (B*T, S, C)
        obs_tokens = obs_tokens.reshape(batch_size, seq_len * num_tokens, -1)
        
        # 2. 时空推演
        h = self.input_proj(obs_tokens)
        for block in self.dynamics_transformer:
            h = block(h)
            
        # 3. 后验分布
        if self.post_net is None:
            self.init_post_net(obs_tokens.size(-1)).to(obs_seq.device)
            
        post_params = self.post_net(torch.cat([h, obs_tokens], dim=-1))
        mu, logvar = torch.chunk(post_params, 2, dim=-1)
        s_t = self.reparameterize(mu, logvar)
        
        return (h, s_t), (mu, logvar)

    def imagine_next(self, prev_states_tokens, action=None):
        """想象下一帧"""
        # prev_states_tokens: (Batch, History_Len * Tokens_Per_Frame, Dim)
        h = self.input_proj(prev_states_tokens)
        for block in self.dynamics_transformer:
            h = block(h)
            
        # 预测下一帧的所有 Tokens
        num_tokens = self.projection.num_tokens
        last_tokens_h = h[:, -num_tokens:, :] # 取最后一步对应的所有空间 tokens
        
        prior_params = self.prior_head(last_tokens_h)
        mu, logvar = torch.chunk(prior_params, 2, dim=-1)
        s_next = self.reparameterize(mu, logvar)
        
        return (last_tokens_h, s_next), (mu, logvar)
