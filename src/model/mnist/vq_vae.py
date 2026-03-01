import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.components.resnet import BasicBlock

class VectorQuantizer(nn.Module):
    """
    VQ-VAE 的核心：矢量量化层
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super(VectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)

    def forward(self, inputs, is_spatial=True):
        if is_spatial:
            # inputs shape: (B, C, H, W) -> (B, H, W, C)
            inputs = inputs.permute(0, 2, 3, 1).contiguous()
        
        flat_input = inputs.view(-1, self.embedding_dim)

        # 计算距离
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embedding.weight.t()))

        # 找到最近的索引
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # 量化
        quantized = torch.matmul(encodings, self.embedding.weight).view(inputs.shape)
        
        # 损失计算
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # 直通估计器
        quantized = inputs + (quantized - inputs).detach()
        
        if is_spatial:
            return quantized.permute(0, 3, 1, 2).contiguous(), loss, encoding_indices
        return quantized, loss, encoding_indices

class BaseVQVAE(nn.Module):
    """VQ-VAE 基类"""
    def __init__(self, num_embeddings=512, embedding_dim=32, commitment_cost=0.25):
        super(BaseVQVAE, self).__init__()
        self.vq_layer = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)

    def encode(self, x):
        raise NotImplementedError

    def decode(self, z):
        raise NotImplementedError

    def forward(self, x):
        z = self.encode(x)
        quantized, vq_loss, _ = self.vq_layer(z, is_spatial=len(z.shape) == 4)
        x_recon = self.decode(quantized)
        return x_recon, vq_loss

class FCVQVAE(BaseVQVAE):
    """全连接 VQ-VAE"""
    def __init__(self, input_dim=784, hidden_dim=400, num_embeddings=512, embedding_dim=32):
        super(FCVQVAE, self).__init__(num_embeddings, embedding_dim)
        self.input_dim = input_dim
        self.encoder_fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
        self.decoder_fc = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def encode(self, x):
        return self.encoder_fc(x.view(-1, self.input_dim))

    def decode(self, z):
        return self.decoder_fc(z).view(-1, 1, 28, 28)

class ConvVQVAE(BaseVQVAE):
    """卷积 VQ-VAE"""
    def __init__(self, in_channels=1, num_hiddens=64, num_embeddings=512, embedding_dim=32):
        super(ConvVQVAE, self).__init__(num_embeddings, embedding_dim)
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(in_channels, num_hiddens//2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(num_hiddens//2, num_hiddens, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(num_hiddens, embedding_dim, kernel_size=3, padding=1)
        )
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(embedding_dim, num_hiddens, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(num_hiddens, num_hiddens//2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(num_hiddens//2, in_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        return self.encoder_conv(x)

    def decode(self, z):
        return self.decoder_conv(z)

class ResNetVQVAE(BaseVQVAE):
    """残差 VQ-VAE"""
    def __init__(self, in_channels=1, num_hiddens=64, num_embeddings=512, embedding_dim=32, block=BasicBlock, num_blocks=[2, 2]):
        super(ResNetVQVAE, self).__init__(num_embeddings, embedding_dim)
        self.in_channels = num_hiddens
        self.block_expansion = block.expansion

        # Encoder
        self.encoder_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, num_hiddens, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, num_hiddens),
            nn.SiLU(inplace=True)
        )
        self.layer1 = self._make_layer(block, num_hiddens, num_blocks[0], stride=2)  # 14x14
        self.layer2 = self._make_layer(block, embedding_dim // self.block_expansion, num_blocks[1], stride=2) # 7x7
        
        # Decoder
        self.in_channels = embedding_dim
        self.layer3 = self._make_layer(block, embedding_dim // self.block_expansion, num_blocks[1], stride=1)
        self.upsample1 = nn.ConvTranspose2d(embedding_dim, num_hiddens, kernel_size=4, stride=2, padding=1)
        
        self.in_channels = num_hiddens
        self.layer4 = self._make_layer(block, num_hiddens, num_blocks[0], stride=1)
        self.upsample2 = nn.ConvTranspose2d(num_hiddens, num_hiddens // 2, kernel_size=4, stride=2, padding=1)

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

    def encode(self, x):
        h = self.encoder_conv1(x)
        h = self.layer1(h)
        return self.layer2(h)

    def decode(self, z):
        h = self.layer3(z)
        h = self.upsample1(h)
        h = self.layer4(h)
        h = self.upsample2(h)
        return self.final_conv(h)
