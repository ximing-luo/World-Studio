import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.components.resnet import ResBlock
from src.model.components.attention import SEBlock, Focus, UnFocus

class BaseSekiroRSSM(nn.Module):
    """Sekiro RSSM 基类"""
    def __init__(self, deterministic_dim=512, stochastic_dim=64, action_dim=2):
        super(BaseSekiroRSSM, self).__init__()
        self.det_dim = deterministic_dim # 确定性维度
        self.stoch_dim = stochastic_dim # 随机性维度
        self.act_dim = action_dim

        # RNN: h_t = f(h_{t-1}, s_{t-1}, a_{t-1})
        self.rnn_cell = nn.GRUCell(stochastic_dim + action_dim, deterministic_dim)

        # Prior: p(s_t | h_t)
        self.prior_net = nn.Sequential(
            nn.Linear(deterministic_dim, deterministic_dim),
            nn.SiLU(),
            nn.Linear(deterministic_dim, stochastic_dim * 2)
        )

    def encode(self, obs):
        raise NotImplementedError

    def decode(self, h, s):
        raise NotImplementedError

    def get_post_net(self, obs_feat_dim):
        # Posterior: q(s_t | h_t, obs_t)
        return nn.Sequential(
            nn.Linear(self.det_dim + obs_feat_dim, self.det_dim),
            nn.SiLU(),
            nn.Linear(self.det_dim, self.stoch_dim * 2)
        )

    def reparameterize(self, mu, logvar):
        logvar = torch.clamp(logvar, min=-20, max=2) 
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def observe(self, obs, prev_state=None, action=None):
        batch_size = obs.size(0)
        if prev_state is None:
            prev_state = self.init_state(batch_size, obs.device)
        
        h_prev, s_prev = prev_state
        
        rnn_input = s_prev
        if action is not None:
            rnn_input = torch.cat([s_prev, action], dim=-1)
        h_t = self.rnn_cell(rnn_input, h_prev)

        obs_feat = self.encode(obs)

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

        prior_params = self.prior_net(h_t)
        mu, logvar = torch.chunk(prior_params, 2, dim=-1)
        s_t = self.reparameterize(mu, logvar)

        return (h_t, s_t), (mu, logvar)

    def init_state(self, batch_size, device):
        h = torch.zeros(batch_size, self.det_dim, device=device)
        s = torch.zeros(batch_size, self.stoch_dim, device=device)
        return h, s

class ConvSekiroRSSM(BaseSekiroRSSM):
    """卷积版 Sekiro RSSM (适用于 128x240)"""
    def __init__(self, in_channels=3, deterministic_dim=512, stochastic_dim=64, action_dim=2):
        super(ConvSekiroRSSM, self).__init__(deterministic_dim, stochastic_dim, action_dim)
        
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, stride=2, padding=1), # 64x120
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), # 32x60
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), # 16x30
            nn.ReLU(),
            # 1x1 卷积压缩冗余通道，保留 16x30 空间信息
            nn.Conv2d(128, 32, 1),
            nn.ReLU(),
            nn.Conv2d(32, 16, 1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.post_net = self.get_post_net(16 * 16 * 30)
        
        self.decoder_conv = nn.Sequential(
            nn.Linear(deterministic_dim + stochastic_dim, 16 * 16 * 30),
            nn.Unflatten(1, (16, 16, 30)),
            nn.ConvTranspose2d(16, 32, 1), 
            nn.ReLU(),
            nn.ConvTranspose2d(32, 128, 1),
            nn.ReLU(),
            # 从 16x30 开始恢复空间分辨率
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), # 32x60
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), # 64x120
            nn.ReLU(),
            nn.ConvTranspose2d(32, in_channels, 4, stride=2, padding=1), # 128x240
            nn.Sigmoid()
        )

    def encode(self, obs):
        return self.encoder_conv(obs)

    def decode(self, h, s):
        return self.decoder_conv(torch.cat([h, s], dim=-1))

class ResNetSekiroRSSM(BaseSekiroRSSM):
    """残差版 Sekiro RSSM (适用于 128x240) - 深度对称架构"""
    def __init__(self, in_channels=3, num_hiddens=64, deterministic_dim=512, stochastic_dim=64, action_dim=13):
        super(ResNetSekiroRSSM, self).__init__(deterministic_dim, stochastic_dim, action_dim)
        self.num_hiddens = num_hiddens

        # Encoder Pipeline: 128x240 -> 8x15 -> Flattened Feature
        self.encoder_conv = nn.Sequential(
            # 直接四倍下采样，保留 128x240 空间信息，优化计算
            Focus(in_channels, num_hiddens, block_size=4),
            self._make_layer(ResBlock, num_hiddens, num_hiddens, num_blocks=4, stride=1),
            self._make_layer(ResBlock, num_hiddens, num_hiddens * 2, num_blocks=6, stride=2),
            self._make_layer(ResBlock, num_hiddens * 2, num_hiddens * 4, num_blocks=4, stride=2),
            # 点卷积压缩通道，保留 8x15 空间信息
            nn.Sequential(
                nn.Conv2d(num_hiddens * 4, 64, 1, bias=False),
                nn.GroupNorm(8, 64),
                nn.SiLU(inplace=True)
            ),
            nn.Flatten()
        )
        
        self.feat_dim = 64 * 8 * 15
        self.post_net = self.get_post_net(self.feat_dim)

        # Decoder Pipeline: 8x15 -> 128x240 -> Sigmoid
        self.fc_z = nn.Linear(deterministic_dim + stochastic_dim, self.feat_dim)
        self.decoder_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            self._make_layer(ResBlock, 64, 128, num_blocks=4, stride=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            self._make_layer(ResBlock, 128, num_hiddens, num_blocks=6, stride=1),
            # 四倍上采样，恢复 128x240 空间分辨率，减少计算
            UnFocus(num_hiddens, in_channels, block_size=4),
            nn.Sigmoid()
        )

    def _make_layer(self, block: ResBlock, in_channels, out_channels, num_blocks, stride):
        layers = []
        # 第一块处理 stride 和通道变化
        layers.append(block(in_channels, out_channels, stride=stride))
        # 后续块保持通道和分辨率不变
        for _ in range(1, num_blocks):
            layers.append(block(out_channels, out_channels, stride=1))
        # 统一添加全局注意力机制
        layers.append(SEBlock(out_channels))
        return nn.Sequential(*layers)

    def encode(self, obs):
        return self.encoder_conv(obs)

    def decode(self, h, s):
        z = torch.cat([h, s], dim=-1)
        h = self.fc_z(z).view(-1, 64, 8, 15)
        return self.decoder_conv(h)
