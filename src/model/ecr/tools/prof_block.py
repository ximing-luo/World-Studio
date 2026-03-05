import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import time
from collections import OrderedDict

# 添加项目根目录到路径
root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(root_path)

from src.model.components.resnet import ResBlock
from src.model.ecr.ecr import (
    CrossScholarFusion, EfficientEvolutionLayer, EfficientCrossResBlock
)

def get_conv2d_flops(module, input_shape):
    h_in, w_in = input_shape[2], input_shape[3]
    k = module.kernel_size[0]
    s = module.stride[0]
    p = module.padding[0]
    h_out = (h_in + 2 * p - k) // s + 1
    w_out = (w_in + 2 * p - k) // s + 1
    flops = 2 * h_out * w_out * module.out_channels * (module.in_channels // module.groups) * k * k
    return flops, (input_shape[0], module.out_channels, h_out, w_out)

def get_linear_flops(module, input_shape):
    flops = 2 * module.in_features * module.out_features * input_shape[0]
    return flops, (input_shape[0], module.out_features)

class LayerMemoryTracker:
    """
    通过 Hook 记录每一层的显存消耗变化
    """
    def __init__(self, model):
        self.model = model
        self.stats = OrderedDict()
        self.hooks = []
        self.device = next(model.parameters()).device

    def _get_pre_hook(self, name):
        def pre_hook(module, input):
            torch.cuda.synchronize(self.device)
            # 记录执行前的显存
            self.stats[name]['mem_before'] = torch.cuda.memory_allocated(self.device)
            # 重置峰值统计，捕获本层执行期间的峰值
            torch.cuda.reset_peak_memory_stats(self.device)
        return pre_hook

    def _get_hook(self, name):
        def hook(module, input, output):
            torch.cuda.synchronize(self.device)
            # 记录执行后的显存
            self.stats[name]['mem_after'] = torch.cuda.memory_allocated(self.device)
            # 记录执行期间的峰值显存
            self.stats[name]['mem_peak'] = torch.cuda.max_memory_allocated(self.device)
        return hook

    def register_hooks(self):
        for name, module in self.model.named_modules():
            # 我们只追踪关键模块或叶子节点
            is_fusion = isinstance(module, CrossScholarFusion)
            if is_fusion or len(list(module.children())) == 0:
                self.stats[name] = {'mem_before': 0, 'mem_after': 0, 'mem_peak': 0, 'type': module.__class__.__name__}
                self.hooks.append(module.register_forward_pre_hook(self._get_pre_hook(name)))
                self.hooks.append(module.register_forward_hook(self._get_hook(name)))

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

class ManualFlopCounter:
    def __init__(self, model):
        self.model = model
        self.layer_stats = OrderedDict()
        self.hooks = []

    def _get_hook(self, name):
        def hook(module, input, output):
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
                    f1 = 2 * input_shape[2] * input_shape[3] * module.latent_dim * module.in_channels
                    f2 = 2 * input_shape[2] * input_shape[3] * module.out_channels * module.latent_dim
                    flops = f1 + f2
                
                if flops > 0:
                    if name not in self.layer_stats:
                        self.layer_stats[name] = {'flops': 0, 'type': module.__class__.__name__}
                    self.layer_stats[name]['flops'] += flops
        return hook

    def register_hooks(self):
        for name, module in self.model.named_modules():
            if isinstance(module, CrossScholarFusion) or len(list(module.children())) == 0:
                self.hooks.append(module.register_forward_hook(self._get_hook(name)))

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

def measure_runtime_memory(model, input_data, iterations=10, is_training=False):
    """
    “纯净实验室”模式：测量显存增量峰值 (Delta Peak)
    """
    device = input_data.device
    model.to(device)
    
    # 1. 彻底清空缓存并记录基准显存 (Model Parameters + Buffers)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)
    base_mem = torch.cuda.memory_allocated(device)
    
    # 2. Warmup: 预热以确保分配 CUDA context 和 kernel，不计入后续测量
    for _ in range(5):
        if is_training:
            model.train()
            model.zero_grad(set_to_none=True)
            out = model(input_data)
            loss = out.mean()
            loss.backward()
            model.zero_grad(set_to_none=True)
        else:
            model.eval()
            with torch.no_grad():
                model(input_data)
    
    # 3. 正式测量
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)
    start_time = time.time()
    
    for _ in range(iterations):
        if is_training:
            model.train()
            model.zero_grad(set_to_none=True) # 严格释放上一轮梯度
            out = model(input_data)
            loss = out.mean()
            loss.backward()
        else:
            model.eval()
            with torch.no_grad():
                model(input_data)
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    # 4. 计算增量峰值
    peak_mem_abs = torch.cuda.max_memory_allocated(device)
    peak_mem_delta_mb = (peak_mem_abs - base_mem) / 1024 / 1024
    avg_time = (end_time - start_time) / iterations * 1000  # ms
    
    return avg_time, peak_mem_delta_mb

def analyze_block(name, module, input_data, is_training=False):
    print(f"\n>>> Analyzing: {name} (Training={is_training})")
    print(f"{'='*120}")
    
    # 核心：根据模式设置训练状态，并开启/关闭梯度计算
    if is_training:
        module.train()
    else:
        module.eval()
    
    # 1. 追踪每一层的显存和算力 (这一步会开启 Hook)
    flop_counter = ManualFlopCounter(module)
    mem_tracker = LayerMemoryTracker(module)
    
    flop_counter.register_hooks()
    mem_tracker.register_hooks()
    
    # 执行一次前向 (仅为了获取每一层的数据)
    # 注意：在训练模式下，这里必须允许梯度流过，否则 Checkpoint 不会触发
    with torch.set_grad_enabled(is_training):
        module.zero_grad(set_to_none=True)
        module(input_data)
    
    print(f"{'Layer Name':<40} | {'Type':<15} | {'FLOPs (M)':<12} | {'Persistent (MB)':<15} | {'Peak (MB)':<12}")
    print(f"{'-'*120}")
    
    total_flops = 0
    for l_name in mem_tracker.stats:
        m_stats = mem_tracker.stats[l_name]
        f_stats = flop_counter.layer_stats.get(l_name, {'flops': 0})
        
        flops_m = f_stats['flops'] / 1e6
        total_flops += f_stats['flops']
        
        # 持久化显存 = 执行后 - 执行前 (表示这一层计算完后，系统依然持有的显存)
        # 注意：在 GC 模式下，对于内部层，这个值理论上应该很小或为 0 (如果能被正确释放)
        persistent_mem_mb = (m_stats['mem_after'] - m_stats['mem_before']) / 1024 / 1024
        # 层内峰值显存 = 峰值 - 执行前 (相对于起始点的增量)
        peak_mem_mb = (m_stats['mem_peak'] - m_stats['mem_before']) / 1024 / 1024
        
        print(f"{l_name:<40} | {m_stats['type']:<15} | {flops_m:<12.4f} | {persistent_mem_mb:<15.4f} | {peak_mem_mb:<12.4f}")
    
    # 核心：在测量整体性能前彻底移除所有 Hook，避免干扰 Checkpoint 释放显存
    flop_counter.remove_hooks()
    mem_tracker.remove_hooks()
    
    # 2. 测量整体性能 (此时环境是“纯净”的)
    avg_time, peak_mem_delta = measure_runtime_memory(module, input_data, is_training=is_training)
    intensity = total_flops / (peak_mem_delta * 1024 * 1024) if peak_mem_delta > 0 else 0
    
    print(f"{'-'*120}")
    print(f"{'BLOCK TOTAL SUMMARY':<40}")
    print(f"{'Total FLOPs':<30} | {total_flops/1e9:<15.4f} GFLOPs")
    print(f"{'Avg Time (per iter)':<30} | {avg_time:<15.4f} ms")
    print(f"{'Overall Peak Delta (MB)':<30} | {peak_mem_delta:<15.4f} MB")
    print(f"{'Arithmetic Intensity':<30} | {intensity:<15.4f} FLOPs/Byte")
    
    return {
        'flops': total_flops,
        'time': avg_time,
        'mem': peak_mem_delta,
        'intensity': intensity
    }

def run_full_comparison():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        print("ERROR: Layer-wise memory tracking requires CUDA.")
        return
        
    BATCH_SIZE = 64
    CHANNELS = 64
    H, W = 32, 60
    # 核心修复：必须开启 requires_grad，Checkpoint 才会生效
    input_data = torch.randn(BATCH_SIZE, CHANNELS, H, W, device=device, requires_grad=True)

    # 1. ResBlock
    res_block = ResBlock(CHANNELS, CHANNELS, stride=1).to(device)
    res_results = analyze_block("ResBlock (Baseline)", res_block, input_data, is_training=True)
    
    # 清理缓存
    torch.cuda.empty_cache()

    # 2. ECR L5 Normal (设置为 5 层)
    ecr_block = EfficientCrossResBlock(CHANNELS, CHANNELS, num_evolve_layers=5, expansion=2, use_checkpoint=False).to(device)
    ecr_results = analyze_block("ECR L5 (Normal Mode)", ecr_block, input_data, is_training=True)
    
    torch.cuda.empty_cache()

    # 3. ECR L5 Gradient Checkpointing (设置为 5 层)
    ecr_block_cp = EfficientCrossResBlock(CHANNELS, CHANNELS, num_evolve_layers=5, expansion=2, use_checkpoint=True).to(device)
    ecr_cp_results = analyze_block("ECR L5 (Checkpoint Mode)", ecr_block_cp, input_data, is_training=True)

    print(f"\n{'='*120}")
    print(f"{'FINAL PERFORMANCE REPORT':^120}")
    print(f"{'='*120}")
    print(f"{'Block Type':<30} | {'GFLOPs':<15} | {'Time (ms)':<15} | {'Mem Delta (MB)':<15} | {'Intensity':<15}")
    print(f"{'-'*120}")
    
    def print_row(name, res):
        print(f"{name:<30} | {res['flops']/1e9:<15.4f} | {res['time']:<15.4f} | {res['mem']:<15.4f} | {res['intensity']:<15.4f}")

    print_row("ResBlock", res_results)
    print_row("ECR L5 (Normal)", ecr_results)
    print_row("ECR L5 (Checkpoint)", ecr_cp_results)
    print(f"{'='*120}\n")

if __name__ == "__main__":
    run_full_comparison()
