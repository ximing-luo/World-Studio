import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
from collections import OrderedDict

# 添加项目根目录到路径
root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(root_path)

from src.model.components.resnet import ResBlock
from src.model.ecr.ecr import (
    CrossScholarFusion, EfficientEvolutionLayer, EfficientCrossResBlock
)

def get_conv2d_flops(module, input_shape):
    # input_shape: (B, C, H, W)
    h_in, w_in = input_shape[2], input_shape[3]
    k = module.kernel_size[0]
    s = module.stride[0]
    p = module.padding[0]
    h_out = (h_in + 2 * p - k) // s + 1
    w_out = (w_in + 2 * p - k) // s + 1
    
    # 2 * MACs
    flops = 2 * h_out * w_out * module.out_channels * (module.in_channels // module.groups) * k * k
    return flops, (input_shape[0], module.out_channels, h_out, w_out)

def get_linear_flops(module, input_shape):
    # input_shape: (B, in_features)
    flops = 2 * module.in_features * module.out_features * input_shape[0]
    return flops, (input_shape[0], module.out_features)

class ManualFlopCounter:
    """
    通过 Hook 记录输入形状，并手动计算常用层的 FLOPs
    """
    def __init__(self, model):
        self.model = model
        self.layer_stats = OrderedDict()
        self.hooks = []

    def _get_hook(self, name):
        def hook(module, input, output):
            # 如果是 CrossScholarFusion，我们将其视为一个整体统计，不进入其内部
            is_fusion = isinstance(module, CrossScholarFusion)
            if is_fusion or len(list(module.children())) == 0:
                input_data = input[0]
                input_shape = input_data.shape
                
                flops = 0
                if isinstance(module, nn.Conv2d):
                    flops, _ = get_conv2d_flops(module, input_shape)
                elif isinstance(module, nn.Linear):
                    flops, _ = get_linear_flops(module, input_shape)
                elif is_fusion:
                    # 手动计算 CrossScholarFusion 的 FLOPs (低秩近似: C_in -> 16 -> C_out)
                    # 1. 投影到潜空间: 2 * H * W * latent * C_in
                    f1 = 2 * input_shape[2] * input_shape[3] * module.latent_dim * module.in_channels
                    # 2. 从潜空间还原: 2 * H * W * C_out * latent
                    f2 = 2 * input_shape[2] * input_shape[3] * module.out_channels * module.latent_dim
                    flops = f1 + f2
                
                if flops > 0:
                    if name not in self.layer_stats:
                        self.layer_stats[name] = {'flops': 0, 'type': module.__class__.__name__}
                    self.layer_stats[name]['flops'] += flops
        return hook

    def register_hooks(self):
        for name, module in self.model.named_modules():
            # 记录 CrossScholarFusion 或叶子节点
            if isinstance(module, CrossScholarFusion) or len(list(module.children())) == 0:
                self.hooks.append(module.register_forward_hook(self._get_hook(name)))

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

def analyze_detailed_layers(name, module, input_data):
    print(f"\nDetailed Layer-wise FLOPs for: {name}")
    print(f"{'-'*100}")
    print(f"{'Layer Name':<50} | {'Layer Type':<20} | {'FLOPs (M)':<15}")
    print(f"{'-'*100}")
    
    counter = ManualFlopCounter(module)
    counter.register_hooks()
    
    with torch.no_grad():
        module.eval()
        module(input_data)
    
    total_layer_flops = 0
    for l_name, stats in counter.layer_stats.items():
        m_flops = stats['flops'] / 1e6
        total_layer_flops += stats['flops']
        print(f"{l_name:<50} | {stats['type']:<20} | {m_flops:<15.4f}")
    
    counter.remove_hooks()
    print(f"{'-'*100}")
    print(f"{'Total Sum of Layers':<50} | {'-':<20} | {total_layer_flops/1e6:<15.4f} M")
    return total_layer_flops

def run_comparison():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BATCH_SIZE = 64
    CHANNELS = 64
    H, W = 32, 60
    INPUT_SHAPE = (BATCH_SIZE, CHANNELS, H, W)
    dummy_input = torch.randn(*INPUT_SHAPE).to(device)

    # 1. ResBlock
    res_block = ResBlock(CHANNELS, CHANNELS, stride=1).to(device).eval()
    
    # 2. ECR L3
    ecr_block = EfficientCrossResBlock(CHANNELS, CHANNELS, num_evolve_layers=1, expansion=2, use_checkpoint=False).to(device).eval()

    print(f"\n{'='*120}")
    print(f"{'SEC L3 vs ResBlock: Layer-wise FLOPs Diagnosis':^120}")
    print(f"{'='*120}")

    # 分析 ResBlock
    res_total = analyze_detailed_layers("ResBlock (Baseline)", res_block, dummy_input)
    
    # 分析 ECR L3
    ecr_total = analyze_detailed_layers("ECR L3 (EfficientCrossResBlock)", ecr_block, dummy_input)

    print(f"\n{'='*120}")
    print(f"{'FINAL SUMMARY':^120}")
    print(f"{'='*120}")
    print(f"ResBlock Total FLOPs: {res_total/1e9:.4f} GFLOPs")
    print(f"ECR L3 Total FLOPs:   {ecr_total/1e9:.4f} GFLOPs")
    print(f"Ratio (ECR/Res):      {ecr_total/res_total:.2f}x")
    print(f"{'='*120}\n")

if __name__ == "__main__":
    run_comparison()
