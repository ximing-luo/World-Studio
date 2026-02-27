import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.dream.vae import ResVAE, VAE


class RSSM(nn.Module):
    """
    Recurrent State Space Model (RSSM)
    将 VAE 的静态表征能力升级为具备时序预测能力的世界模型。
    状态组成：
    - h_t: 确定性状态 (Deterministic state, GRU 隐藏层)
    - s_t: 随机性状态 (Stochastic state, VAE 采样结果)
    """
    def __init__(self, backbone, deterministic_dim=256, stochastic_dim=32, action_dim=0):
        super(RSSM, self).__init__()
        self.backbone = backbone
        self.det_dim = deterministic_dim
        self.stoch_dim = stochastic_dim
        self.act_dim = action_dim

        # 判断 backbone 类型
        self.is_resvae = isinstance(backbone, ResVAE)
        
        # 1. 确定性状态转移 (RNN): (h_{t-1}, s_{t-1}, a_{t-1}) -> h_t
        self.rnn_cell = nn.GRUCell(stochastic_dim + action_dim, deterministic_dim)

        # 2. 先验模型 (Prior): h_t -> p(s_t | h_t)
        self.prior_net = nn.Sequential(
            nn.Linear(deterministic_dim, deterministic_dim),
            nn.SiLU(),
            nn.Linear(deterministic_dim, stochastic_dim * 2) # mu and logvar
        )

        # 根据 backbone 确定观测特征维度
        if self.is_resvae:
            # ResVAE: 128 * expansion * 7 * 7
            obs_feat_dim = 128 * backbone.block_expansion * 7 * 7
            # 状态到隐空间映射维度
            self.state_to_latent_dim = obs_feat_dim
        else:
            # VAE (MLP): hidden_dim (e.g. 400)
            # 我们使用 VAE 的 fc1 输出作为观测特征
            obs_feat_dim = backbone.fc1.out_features
            # 状态到隐空间映射维度 -> 对应 VAE 的 latent_dim
            # 因为 VAE.decode 接受 latent_dim 大小的输入
            self.state_to_latent_dim = backbone.fc21.out_features # latent_dim

        # 3. 后验模型 (Posterior): (h_t, obs_t) -> q(s_t | h_t, obs_t)
        self.post_net = nn.Sequential(
            nn.Linear(deterministic_dim + obs_feat_dim, deterministic_dim),
            nn.SiLU(),
            nn.Linear(deterministic_dim, stochastic_dim * 2)
        )

        # 4. 解码器重定义: 从 (h_t, s_t) 映射回像素
        # 我们修改 VAE 的解码路径，使其接受 (h_t, s_t) 的组合
        self.state_to_latent = nn.Linear(deterministic_dim + stochastic_dim, self.state_to_latent_dim)

    def observe(self, obs, prev_state=None, action=None):
        """
        根据观测值更新状态 (Posterior Step)
        obs: (B, C, H, W)
        """
        batch_size = obs.size(0)
        if prev_state is None:
            prev_state = self.init_state(batch_size, obs.device)
        
        h_prev, s_prev = prev_state
        
        # 1. RNN 更新确定性状态
        rnn_input = s_prev
        if action is not None:
            rnn_input = torch.cat([s_prev, action], dim=-1)
        h_t = self.rnn_cell(rnn_input, h_prev)

        # 2. 提取当前观测特征
        if self.is_resvae:
            obs_feat = self.backbone.encoder_conv1(obs)
            obs_feat = self.backbone.layer1(obs_feat)
            obs_feat = self.backbone.layer2(obs_feat)
            obs_feat = obs_feat.view(batch_size, -1)
        else:
            # MLP VAE
            flat_obs = obs.view(batch_size, -1)
            obs_feat = F.relu(self.backbone.fc1(flat_obs))

        # 3. 计算后验分布并采样
        post_params = self.post_net(torch.cat([h_t, obs_feat], dim=-1))
        mu, logvar = torch.chunk(post_params, 2, dim=-1)
        s_t = self.backbone.reparameterize(mu, logvar)

        return (h_t, s_t), (mu, logvar)

    def imagine(self, prev_state, action=None):
        """
        纯梦境预测，不依赖观测 (Prior Step)
        """
        h_prev, s_prev = prev_state
        
        rnn_input = s_prev
        if action is not None:
            rnn_input = torch.cat([s_prev, action], dim=-1)
        h_t = self.rnn_cell(rnn_input, h_prev)

        # 纯先验预测
        prior_params = self.prior_net(h_t)
        mu, logvar = torch.chunk(prior_params, 2, dim=-1)
        s_t = self.backbone.reparameterize(mu, logvar)

        return (h_t, s_t), (mu, logvar)

    def decode(self, state):
        """
        将 RSSM 状态还原为图像
        """
        h_t, s_t = state
        latent = self.state_to_latent(torch.cat([h_t, s_t], dim=-1))
        
        if self.is_resvae:
            # ResVAE 的解码逻辑需要 spatial feature map
            h = latent.view(latent.size(0), 128 * self.backbone.block_expansion, 7, 7)
            h = self.backbone.layer3(h)
            h = self.backbone.upsample1(h)
            h = self.backbone.layer4(h)
            h = self.backbone.upsample2(h)
            return self.backbone.final_conv(h)
        else:
            # MLP VAE 的解码逻辑直接接受 latent vector
            return self.backbone.decode(latent).view(-1, 1, 28, 28)

    def init_state(self, batch_size, device):
        h = torch.zeros(batch_size, self.det_dim, device=device)
        s = torch.zeros(batch_size, self.stoch_dim, device=device)
        return h, s
