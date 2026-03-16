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

from src.model.components.resnet import BottleNeck
from src.model.ecr.ecr import (
    CrossScholarFusion, EfficientEvolutionLayer, EfficientCrossResBlock
)
from src.model.ecr.cuda_evolution.ops_evolution import EvolutionLayer
from src.model.ecr.utils.prof_utils import ManualFlopCounter

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
            is_evolution = isinstance(module, EvolutionLayer)
            if is_fusion or is_evolution or len(list(module.children())) == 0:
                self.stats[name] = {'mem_before': 0, 'mem_after': 0, 'mem_peak': 0, 'type': module.__class__.__name__}
                self.hooks.append(module.register_forward_pre_hook(self._get_pre_hook(name)))
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
    
    # 核心：根据模式设置训练状态
    if is_training:
        module.train()
    else:
        module.eval()
    
    # 1. 静态算力分析 (不干扰显存)
    flop_counter = ManualFlopCounter(module)
    flop_counter.register_hooks()
    
    # 2. 运行时显存快照 (Timeline Snapshot)
    device = input_data.device
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    # 快照 1: 基准 (模型 + 输入)
    base_mem = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    
    # 执行前向
    with torch.set_grad_enabled(is_training):
        module.zero_grad(set_to_none=True)
        output = module(input_data)
        
        torch.cuda.synchronize()
        # 快照 2: 前向结束后的留存 (这是验证 GC 的关键点)
        post_forward_mem = torch.cuda.memory_allocated(device)
        # 快照 3: 过程中的最高峰值
        peak_forward_mem = torch.cuda.max_memory_allocated(device)
        
        if is_training:
            # 执行反向
            loss = output.mean()
            loss.backward()
            torch.cuda.synchronize()
            # 快照 4: 反向结束后的留存 (应该包含梯度)
            post_backward_mem = torch.cuda.memory_allocated(device)
            peak_total_mem = torch.cuda.max_memory_allocated(device)
        else:
            post_backward_mem = post_forward_mem
            peak_total_mem = peak_forward_mem

    # 移除算力钩子
    flop_counter.remove_hooks()
    
    # 计算统计数据 (MB)
    def to_mb(x): return x / 1024 / 1024
    
    retained_act = to_mb(post_forward_mem - base_mem)
    peak_act = to_mb(peak_forward_mem - base_mem)
    grad_mem = to_mb(post_backward_mem - post_forward_mem) if is_training else 0
    
    total_flops = sum(s['flops'] for s in flop_counter.layer_stats.values())
    
    print(f"{'Metric':<40} | {'Value':<20}")
    print(f"{'-'*65}")
    print(f"{'Total FLOPs':<40} | {total_flops/1e9:<10.4f} GFLOPs")
    print(f"{'Base Memory (Model+Input)':<40} | {to_mb(base_mem):<10.2f} MB")
    print(f"{'Peak Forward Memory (Working)':<40} | {peak_act:<10.2f} MB")
    print(f"{'Retained Forward Memory (Activations)':<40} | {retained_act:<10.2f} MB")
    if is_training:
        print(f"{'Retained Backward Memory (Gradients)':<40} | {grad_mem:<10.2f} MB")
        print(f"{'Overall Peak Memory':<40} | {to_mb(peak_total_mem - base_mem):<10.2f} MB")
    
    print(f"{'-'*120}")
    
    # 核心验证逻辑
    if "Checkpoint" in name:
        # 在 GC 模式下，留存的激活值应该只占 5 层总和的极小部分 (约 2 层的大小：输入+输出)
        print(f"VERIFICATION: Checkpoint mode should have low 'Retained Forward Memory'.")
        print(f"Current Retained: {retained_act:.2f} MB")
    
    return {
        'flops': total_flops,
        'mem': to_mb(peak_total_mem - base_mem),
        'retained': retained_act
    }

def run_full_comparison():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        print("ERROR: Layer-wise memory tracking requires CUDA.")
        return
        
    BATCH_SIZE = 64
    CHANNELS = 256 # 高通道
    H, W = 16, 16  # 低分辨率
    # 核心修复：必须开启 requires_grad
    input_data = torch.randn(BATCH_SIZE, CHANNELS, H, W, device=device, requires_grad=True)

    # 1. BottleNeck (通道数不变)
    bottleneck_block = BottleNeck(CHANNELS, CHANNELS, stride=1).to(device)
    bottleneck_results = analyze_block("BottleNeck (Baseline)", bottleneck_block, input_data, is_training=True)
    
    # 清理缓存
    torch.cuda.empty_cache()

    # 2. ECR V8 Normal (默认 expansion=4)
    ecr_block = EfficientCrossResBlock(CHANNELS, CHANNELS, expansion=4, stride=1).to(device)
    ecr_results = analyze_block("ECR V8 (Normal Mode)", ecr_block, input_data, is_training=True)
    
    print(f"\n{'='*120}")
    print(f"{'FINAL PERFORMANCE REPORT':^120}")
    print(f"{'='*120}")
    print(f"{'Block Type':<30} | {'GFLOPs':<15} | {'Peak Mem (MB)':<20} | {'Retained Act (MB)':<20}")
    print(f"{'-'*120}")
    
    def print_row(name, res):
        print(f"{name:<30} | {res['flops']/1e9:<15.4f} | {res['mem']:<20.2f} | {res['retained']:<20.2f}")

    print_row("BottleNeck", bottleneck_results)
    print_row("ECR V8 (Normal)", ecr_results)
    print(f"{'='*120}")

if __name__ == "__main__":
    run_full_comparison()
