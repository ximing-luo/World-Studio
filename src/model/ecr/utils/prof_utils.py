import torch
import torch.nn as nn
from collections import OrderedDict
from src.model.ecr.ecr import CrossScholarFusion
from src.model.ecr.cuda_evolution.ops_evolution import EvolutionLayer

def get_conv2d_flops(module, input_shape):
    """计算 Conv2d 的 FLOPs (2 * MACs)"""
    h_in, w_in = input_shape[2], input_shape[3]
    k = module.kernel_size[0]
    s = module.stride[0]
    p = module.padding[0]
    h_out = (h_in + 2 * p - k) // s + 1
    w_out = (w_in + 2 * p - k) // s + 1
    flops = 2 * h_out * w_out * module.out_channels * (module.in_channels // module.groups) * k * k
    return flops, (input_shape[0], module.out_channels, h_out, w_out)

def get_linear_flops(module, input_shape):
    """计算 Linear 的 FLOPs (2 * MACs)"""
    flops = 2 * module.in_features * module.out_features * input_shape[0]
    return flops, (input_shape[0], module.out_features)

class ManualFlopCounter:
    """
    通过 Hook 记录每一层的算力消耗
    """
    def __init__(self, model):
        self.model = model
        self.layer_stats = OrderedDict()
        self.hooks = []

    def _get_hook(self, name):
        def hook(module, input, output):
            is_fusion = isinstance(module, CrossScholarFusion)
            is_evolution = isinstance(module, EvolutionLayer)
            if is_fusion or is_evolution or len(list(module.children())) == 0:
                input_data = input[0]
                input_shape = input_data.shape
                
                flops = 0
                if isinstance(module, nn.Conv2d):
                    flops, _ = get_conv2d_flops(module, input_shape)
                elif isinstance(module, nn.Linear):
                    flops, _ = get_linear_flops(module, input_shape)
                elif is_fusion:
                    # 手动计算 CrossScholarFusion 的 FLOPs (低秩近似)
                    f1 = 2 * input_shape[2] * input_shape[3] * module.latent_dim * module.in_channels
                    f2 = 2 * input_shape[2] * input_shape[3] * module.out_channels * module.latent_dim
                    flops = f1 + f2
                elif is_evolution:
                    # EvolutionLayer: 1 层 3x3 DW 卷积 (融合算子内部)
                    # FLOPs = 1层 * 2 (MACs to FLOPs) * (3*3 kernel) * channels * (H * W)
                    flops = 1 * 2 * 9 * module.channels * input_shape[2] * input_shape[3]
                
                if flops > 0:
                    if name not in self.layer_stats:
                        self.layer_stats[name] = {'flops': 0, 'type': module.__class__.__name__}
                    self.layer_stats[name]['flops'] += flops
        return hook

    def register_hooks(self):
        for name, module in self.model.named_modules():
            if isinstance(module, (CrossScholarFusion, EvolutionLayer)) or len(list(module.children())) == 0:
                self.hooks.append(module.register_forward_hook(self._get_hook(name)))

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
