import torch
import torch.nn as nn

# 改造后的 SEBlock -> PixelGatedBlock (类似于视觉版的 SwiGLU)
class PixelGatedBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # 用 1x1 卷积产生两个支路：一个做“值”，一个做“门”
        self.gate_conv = nn.Conv2d(channels, channels, 1)
        self.value_conv = nn.Conv2d(channels, channels, 1)
        
    def forward(self, x):
        # 模仿 SwiGLU: SiLU(支路1) * 支路2
        gate = torch.sigmoid(self.gate_conv(x)) # 或者用 SiLU
        value = self.value_conv(x)
        return value * gate