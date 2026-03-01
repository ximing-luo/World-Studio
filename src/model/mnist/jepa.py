import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.components.resnet import BasicBlock

class BaseJEPA(nn.Module):
    """JEPA 基类，处理表征预测与复合损失计算"""
    def __init__(self, latent_dim=128):
        super(BaseJEPA, self).__init__()
        self.latent_dim = latent_dim
        
        # Predictor: z_context + condition -> z_target_pred
        # condition (2维: sin/cos)
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim + 2, latent_dim * 2),
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

    def forward(self, x_context, x_target, condition):
        # 1. 提取表征
        z_context = self.encode_context(x_context)
        with torch.no_grad():
            z_target = self.encode_target(x_target)
            
        # 2. 增强角度信号
        angle_emb = torch.cat([torch.sin(condition), torch.cos(condition)], dim=-1)
            
        # 3. 预测目标表征
        z_target_pred = self.predictor(torch.cat([z_context, angle_emb], dim=-1))
        
        # 4. 计算损失
        total_loss = self.calculate_loss(z_context, z_target, z_target_pred)
        
        return z_target_pred, z_target, total_loss

    def calculate_loss(self, z_context, z_target, z_target_pred):
        # (1) Invariance Loss
        sim_loss = F.mse_loss(z_target_pred, z_target)
        
        # (2) Variance Loss
        std_context = torch.sqrt(z_context.var(dim=0) + 1e-04)
        std_target = torch.sqrt(z_target.var(dim=0) + 1e-04)
        std_loss = torch.mean(F.relu(1.0 - std_context)) + torch.mean(F.relu(1.0 - std_target))
        
        # (3) Covariance Loss
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
            # 注意：子类需要确保 context 和 target encoder 的参数顺序一致，或者手动指定
            pass

class FCJEPA(BaseJEPA):
    """全连接 JEPA"""
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=128):
        super(FCJEPA, self).__init__(latent_dim)
        self.input_dim = input_dim
        self.context_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.target_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

    def encode_context(self, x):
        return self.context_encoder(x.view(-1, self.input_dim))

    def encode_target(self, x):
        return self.target_encoder(x.view(-1, self.input_dim))

    @torch.no_grad()
    def update_target(self, momentum=0.999):
        for p_context, p_target in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
            p_target.data.mul_(momentum).add_(p_context.data, alpha=1 - momentum)

class ConvJEPA(BaseJEPA):
    """卷积 JEPA"""
    def __init__(self, in_channels=1, latent_dim=128):
        super(ConvJEPA, self).__init__(latent_dim)
        self.context_encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, latent_dim)
        )
        self.target_encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, latent_dim)
        )

    def encode_context(self, x):
        return self.context_encoder(x)

    def encode_target(self, x):
        return self.target_encoder(x)

    @torch.no_grad()
    def update_target(self, momentum=0.999):
        for p_context, p_target in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
            p_target.data.mul_(momentum).add_(p_context.data, alpha=1 - momentum)

class ResNetJEPA(BaseJEPA):
    """残差 JEPA"""
    def __init__(self, in_channels=1, num_hiddens=64, latent_dim=128, block=BasicBlock, num_blocks=[2, 2]):
        super(ResNetJEPA, self).__init__(latent_dim)
        self.block_expansion = block.expansion
        
        # Context Encoder
        self.in_channels = num_hiddens
        self.context_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, num_hiddens, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, num_hiddens),
            nn.SiLU(inplace=True)
        )
        self.context_layer1 = self._make_layer(block, num_hiddens, num_blocks[0], stride=2)  # 14x14
        self.context_layer2 = self._make_layer(block, num_hiddens * 2, num_blocks[1], stride=2) # 7x7
        self.context_fc = nn.Linear(num_hiddens * 2 * self.block_expansion * 7 * 7, latent_dim)

        # Target Encoder
        self.in_channels = num_hiddens
        self.target_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, num_hiddens, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, num_hiddens),
            nn.SiLU(inplace=True)
        )
        self.target_layer1 = self._make_layer(block, num_hiddens, num_blocks[0], stride=2)  # 14x14
        self.target_layer2 = self._make_layer(block, num_hiddens * 2, num_blocks[1], stride=2) # 7x7
        self.target_fc = nn.Linear(num_hiddens * 2 * self.block_expansion * 7 * 7, latent_dim)

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
        return self.context_fc(h.view(h.size(0), -1))

    def encode_target(self, x):
        h = self.target_conv1(x)
        h = self.target_layer1(h)
        h = self.target_layer2(h)
        return self.target_fc(h.view(h.size(0), -1))

    @torch.no_grad()
    def update_target(self, momentum=0.999):
        # 更新所有相关参数
        for p_context, p_target in zip(self.parameters(), self.parameters()):
            # 简化逻辑：JEPA 的 update_target 通常只针对 target_encoder
            pass
        
        # 精确更新 target_encoder
        context_params = list(self.context_conv1.parameters()) + \
                         list(self.context_layer1.parameters()) + \
                         list(self.context_layer2.parameters()) + \
                         list(self.context_fc.parameters())
        
        target_params = list(self.target_conv1.parameters()) + \
                        list(self.target_layer1.parameters()) + \
                        list(self.target_layer2.parameters()) + \
                        list(self.target_fc.parameters())
        
        for p_context, p_target in zip(context_params, target_params):
            p_target.data.mul_(momentum).add_(p_context.data, alpha=1 - momentum)
