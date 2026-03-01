import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.components.resnet import BasicBlock

class BaseRSSM(nn.Module):
    """
    RSSM 基类，定义状态转移和核心逻辑
    状态：h_t (确定性), s_t (随机性)
    """
    def __init__(self, deterministic_dim=256, stochastic_dim=32, action_dim=0):
        super(BaseRSSM, self).__init__()
        self.det_dim = deterministic_dim
        self.stoch_dim = stochastic_dim
        self.act_dim = action_dim

        # 1. 确定性状态转移 (RNN)
        self.rnn_cell = nn.GRUCell(stochastic_dim + action_dim, deterministic_dim)

        # 2. 先验模型 (Prior): h_t -> p(s_t | h_t)
        self.prior_net = nn.Sequential(
            nn.Linear(deterministic_dim, deterministic_dim),
            nn.SiLU(),
            nn.Linear(deterministic_dim, stochastic_dim * 2)
        )

    def encode_obs(self, obs):
        raise NotImplementedError

    def decode_state(self, h, s):
        raise NotImplementedError

    def get_post_net(self, obs_feat_dim):
        # 3. 后验模型 (Posterior): (h_t, obs_t) -> q(s_t | h_t, obs_t)
        return nn.Sequential(
            nn.Linear(self.det_dim + obs_feat_dim, self.det_dim),
            nn.SiLU(),
            nn.Linear(self.det_dim, self.stoch_dim * 2)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def observe(self, obs, prev_state=None, action=None):
        batch_size = obs.size(0)
        if prev_state is None:
            prev_state = self.init_state(batch_size, obs.device)
        
        h_prev, s_prev = prev_state
        
        # RNN 更新
        rnn_input = s_prev
        if action is not None:
            rnn_input = torch.cat([s_prev, action], dim=-1)
        h_t = self.rnn_cell(rnn_input, h_prev)

        # 提取观测特征
        obs_feat = self.encode_obs(obs)

        # 后验采样
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

    def init_state(self, batch_size, device):
        h = torch.zeros(batch_size, self.det_dim, device=device)
        s = torch.zeros(batch_size, self.stoch_dim, device=device)
        return h, s

class FCRSSM(BaseRSSM):
    """全连接 RSSM"""
    def __init__(self, input_dim=784, hidden_dim=400, deterministic_dim=256, stochastic_dim=32, action_dim=0):
        super(FCRSSM, self).__init__(deterministic_dim, stochastic_dim, action_dim)
        self.input_dim = input_dim
        self.encoder_fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.post_net = self.get_post_net(hidden_dim)
        self.decoder_fc = nn.Sequential(
            nn.Linear(deterministic_dim + stochastic_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def encode_obs(self, obs):
        return self.encoder_fc(obs.view(-1, self.input_dim))

    def decode_state(self, h, s):
        return self.decoder_fc(torch.cat([h, s], dim=-1)).view(-1, 1, 28, 28)

class ConvRSSM(BaseRSSM):
    """卷积 RSSM"""
    def __init__(self, in_channels=1, deterministic_dim=256, stochastic_dim=32, action_dim=0):
        super(ConvRSSM, self).__init__(deterministic_dim, stochastic_dim, action_dim)
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.post_net = self.get_post_net(64 * 7 * 7)
        self.decoder_conv = nn.Sequential(
            nn.Linear(deterministic_dim + stochastic_dim, 64 * 7 * 7),
            nn.Unflatten(1, (64, 7, 7)),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, in_channels, 4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def encode_obs(self, obs):
        return self.encoder_conv(obs)

    def decode_state(self, h, s):
        return self.decoder_conv(torch.cat([h, s], dim=-1))

class ResNetRSSM(BaseRSSM):
    """残差 RSSM"""
    def __init__(self, in_channels=1, num_hiddens=64, deterministic_dim=256, stochastic_dim=32, action_dim=0, block=BasicBlock, num_blocks=[2, 2]):
        super(ResNetRSSM, self).__init__(deterministic_dim, stochastic_dim, action_dim)
        self.num_hiddens = num_hiddens
        self.in_channels = num_hiddens
        self.block_expansion = block.expansion

        # Encoder
        self.encoder_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, num_hiddens, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, num_hiddens),
            nn.SiLU(inplace=True)
        )
        self.layer1 = self._make_layer(block, num_hiddens, num_blocks[0], stride=2)  # 14x14
        self.layer2 = self._make_layer(block, num_hiddens * 2, num_blocks[1], stride=2) # 7x7
        
        self.post_net = self.get_post_net(num_hiddens * 2 * self.block_expansion * 7 * 7)

        # Decoder
        self.fc_z = nn.Linear(deterministic_dim + stochastic_dim, num_hiddens * 2 * self.block_expansion * 7 * 7)
        self.in_channels = num_hiddens * 2 * self.block_expansion
        self.layer3 = self._make_layer(block, num_hiddens * 2, num_blocks[1], stride=1)
        self.upsample1 = nn.ConvTranspose2d(num_hiddens * 2 * self.block_expansion, num_hiddens * self.block_expansion, kernel_size=2, stride=2)
        
        self.in_channels = num_hiddens * self.block_expansion
        self.layer4 = self._make_layer(block, num_hiddens, num_blocks[0], stride=1)
        self.upsample2 = nn.ConvTranspose2d(num_hiddens * self.block_expansion, num_hiddens // 2, kernel_size=2, stride=2)

        self.final_conv = nn.Sequential(
            nn.Conv2d(num_hiddens // 2, in_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def _make_layer(self, block, out_channels, num_block, stride):
        strides = [stride] + [1] * (num_block - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, stride=s))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def encode_obs(self, obs):
        h = self.encoder_conv1(obs)
        h = self.layer1(h)
        h = self.layer2(h)
        return h.view(h.size(0), -1)

    def decode_state(self, h, s):
        z = torch.cat([h, s], dim=-1)
        h = self.fc_z(z).view(-1, self.num_hiddens * 2 * self.block_expansion, 7, 7)
        h = self.layer3(h)
        h = self.upsample1(h)
        h = self.layer4(h)
        h = self.upsample2(h)
        return self.final_conv(h)
