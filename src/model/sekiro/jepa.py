import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.components.resnet import BasicBlock

class BaseSekiroJEPA(nn.Module):
    """Sekiro JEPA 基类"""
    def __init__(self, latent_dim=256, action_dim=2):
        super(BaseSekiroJEPA, self).__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        
        # Predictor: z_context + action -> z_target_pred
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim + action_dim, latent_dim * 2),
            nn.LayerNorm(latent_dim * 2),
            nn.SiLU(),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim)
        )

    def encode_context(self, x):
        raise NotImplementedError

    def encode_target(self, x):
        raise NotImplementedError

    def forward(self, x_context, x_target, action):
        z_context = self.encode_context(x_context)
        with torch.no_grad():
            z_target = self.encode_target(x_target)
            
        z_target_pred = self.predictor(torch.cat([z_context, action], dim=-1))
        
        # 计算损失 (VicReg 风格)
        total_loss = self.calculate_loss(z_context, z_target, z_target_pred)
        
        return z_target_pred, z_target, total_loss

    def calculate_loss(self, z_context, z_target, z_target_pred):
        sim_loss = F.mse_loss(z_target_pred, z_target)
        
        std_context = torch.sqrt(z_context.var(dim=0) + 1e-04)
        std_target = torch.sqrt(z_target.var(dim=0) + 1e-04)
        std_loss = torch.mean(F.relu(1.0 - std_context)) + torch.mean(F.relu(1.0 - std_target))
        
        def off_diagonal(x):
            n, m = x.shape
            assert n == m
            return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

        c = (z_context.T @ z_context) / (z_context.shape[0] - 1)
        cov_loss = off_diagonal(c).pow_(2).sum() / z_context.shape[1]

        return sim_loss + 1.0 * std_loss + 0.01 * cov_loss

    @torch.no_grad()
    def update_target(self, momentum=0.999):
        """EMA 更新目标编码器"""
        for p_context, p_target in zip(self.parameters(), self.parameters()):
            # 子类需具体实现
            pass

class ConvSekiroJEPA(BaseSekiroJEPA):
    """卷积版 Sekiro JEPA"""
    def __init__(self, in_channels=3, latent_dim=256, action_dim=2, input_res=(128, 240)):
        super(ConvSekiroJEPA, self).__init__(latent_dim, action_dim)
        
        # 计算经过 3 层 stride=2 卷积后的特征图尺寸
        # 128 -> 64 -> 32 -> 16
        # 240 -> 120 -> 60 -> 30
        feat_h = input_res[0] // 8
        feat_w = input_res[1] // 8
        self.flat_dim = 128 * feat_h * feat_w

        self.context_encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(self.flat_dim, latent_dim)
        )
        
        self.target_encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(self.flat_dim, latent_dim)
        )

    def encode_context(self, x):
        return self.context_encoder(x)

    def encode_target(self, x):
        return self.target_encoder(x)

    @torch.no_grad()
    def update_target(self, momentum=0.999):
        for p_context, p_target in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
            p_target.data.mul_(momentum).add_(p_context.data, alpha=1 - momentum)

class ResNetSekiroJEPA(BaseSekiroJEPA):
    """残差版 Sekiro JEPA"""
    def __init__(self, in_channels=3, num_hiddens=64, latent_dim=256, action_dim=2, block=BasicBlock, num_blocks=[2, 2, 2], input_res=(128, 240)):
        super(ResNetSekiroJEPA, self).__init__(latent_dim, action_dim)
        self.block_expansion = block.expansion
        
        # 计算特征图尺寸
        feat_h = input_res[0] // 8
        feat_w = input_res[1] // 8
        self.flat_dim = num_hiddens * 4 * self.block_expansion * feat_h * feat_w

        # Context Encoder
        self.in_channels = num_hiddens
        self.context_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, num_hiddens, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, num_hiddens),
            nn.SiLU(inplace=True)
        )
        self.context_layer1 = self._make_layer(block, num_hiddens, num_blocks[0], stride=2)
        self.context_layer2 = self._make_layer(block, num_hiddens * 2, num_blocks[1], stride=2)
        self.context_layer3 = self._make_layer(block, num_hiddens * 4, num_blocks[2], stride=2)
        self.context_fc = nn.Linear(self.flat_dim, latent_dim)

        # Target Encoder
        self.in_channels = num_hiddens
        self.target_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, num_hiddens, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, num_hiddens),
            nn.SiLU(inplace=True)
        )
        self.target_layer1 = self._make_layer(block, num_hiddens, num_blocks[0], stride=2)
        self.target_layer2 = self._make_layer(block, num_hiddens * 2, num_blocks[1], stride=2)
        self.target_layer3 = self._make_layer(block, num_hiddens * 4, num_blocks[2], stride=2)
        self.target_fc = nn.Linear(self.flat_dim, latent_dim)

    def _make_layer(self, block, out_channels, num_block, stride):
        strides = [stride] + [1] * (num_block - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, stride=s))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def encode_context(self, x):
        h = self.context_conv1(x)
        h = self.context_layer1(h)
        h = self.context_layer2(h)
        h = self.context_layer3(h)
        return self.context_fc(h.view(h.size(0), -1))

    def encode_target(self, x):
        h = self.target_conv1(x)
        h = self.target_layer1(h)
        h = self.target_layer2(h)
        h = self.target_layer3(h)
        return self.target_fc(h.view(h.size(0), -1))

    @torch.no_grad()
    def update_target(self, momentum=0.999):
        context_params = list(self.context_conv1.parameters()) + \
                         list(self.context_layer1.parameters()) + \
                         list(self.context_layer2.parameters()) + \
                         list(self.context_layer3.parameters()) + \
                         list(self.context_fc.parameters())
        
        target_params = list(self.target_conv1.parameters()) + \
                        list(self.target_layer1.parameters()) + \
                        list(self.target_layer2.parameters()) + \
                        list(self.target_layer3.parameters()) + \
                        list(self.target_fc.parameters())
        
        for p_context, p_target in zip(context_params, target_params):
            p_target.data.mul_(momentum).add_(p_context.data, alpha=1 - momentum)
