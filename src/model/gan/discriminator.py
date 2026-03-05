import torch
import torch.nn as nn

class DiscBlock(nn.Module):
    """判别器基础块：卷积 -> 归一化 -> 激活"""
    def __init__(self, in_channels, out_channels, stride=2, use_norm=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, 
                              stride=stride, padding=1, bias=not use_norm)
        self.norm = nn.BatchNorm2d(out_channels) if use_norm else nn.Identity()
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))

class PatchGANDiscriminator(nn.Module):
    """
    加强版 PatchGAN 判别器
    """
    def __init__(self, input_nc=3, ndf=128): # 基础通道数从 64 提升到 128
        super().__init__()
        
        # 1. 初始层：(B, 3, 128, 240) -> (B, 128, 64, 120)
        self.layer1 = DiscBlock(input_nc, ndf, stride=2, use_norm=False)
        
        # 2. 中间下采样：(B, 128, 64, 120) -> (B, 256, 32, 60)
        self.layer2 = DiscBlock(ndf, ndf * 2, stride=2)
        
        # 3. 再次下采样：(B, 256, 32, 60) -> (B, 512, 16, 30)
        self.layer3 = DiscBlock(ndf * 2, ndf * 4, stride=2)
        
        # 4. 深度特征提取：(B, 512, 16, 30) -> (B, 1024, 15, 29)
        self.layer4 = DiscBlock(ndf * 4, ndf * 8, stride=1)
        
        # 5. 最终投影
        self.final = nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.final(x)

class GANLoss(nn.Module):
    """
    对抗损失 (GAN Loss) 辅助类
    支持 LSGAN (MSE) 和标准 GAN (BCE) 模式。
    """
    def __init__(self, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.loss = nn.MSELoss() # 默认使用 LSGAN，训练更稳定

    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        return self.loss(prediction, target_tensor)
