import torch
import torch.nn as nn
import sys
import os
import time
from collections import OrderedDict

# 添加项目根目录到路径
root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(root_path)

from src.model.components.resnet import ResBlock
from src.model.ecr.ecr import EfficientCrossResBlock

def profile_layers(name, model, input_data):
    print(f"\n>>> Profiling: {name}")
    print("=" * 110)
    print(f"{'Layer Name':<45} | {'Type':<20} | {'Mem (MB)':<12} | {'Time (ms)':<10}")
    print("-" * 110)
    
    device = input_data.device
    model.to(device)
    model.train()
    
    # 预热
    with torch.no_grad():
        for _ in range(5):
            _ = model(input_data)
    
    layer_stats = OrderedDict()
    
    def get_pre_hook(name):
        def hook(module, input):
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            mem_before = torch.cuda.memory_allocated(device) / (1024 ** 2)
            layer_stats[name] = {
                'start_event': start_event,
                'mem_before': mem_before,
                'type': module.__class__.__name__
            }
        return hook

    def get_post_hook(name):
        def hook(module, input, output):
            torch.cuda.synchronize()
            end_event = torch.cuda.Event(enable_timing=True)
            end_event.record()
            mem_after = torch.cuda.memory_allocated(device) / (1024 ** 2)
            
            # 等待计时结束
            torch.cuda.synchronize()
            elapsed_time = layer_stats[name]['start_event'].elapsed_time(end_event)
            
            layer_stats[name]['mem_after'] = mem_after
            layer_stats[name]['time'] = elapsed_time
        return hook

    hooks = []
    for n, m in model.named_modules():
        if len(list(m.children())) == 0 or isinstance(m, (nn.Conv2d, nn.Linear, nn.LayerNorm, nn.GroupNorm)):
            hooks.append(m.register_forward_pre_hook(get_pre_hook(n)))
            hooks.append(m.register_forward_hook(get_post_hook(n)))

    # 执行一次前向传播
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    base_mem = torch.cuda.memory_allocated(device) / (1024 ** 2)
    
    with torch.set_grad_enabled(True):
        output = model(input_data)
        loss = output.sum()
        loss.backward()

    # 打印统计结果
    for name, stat in layer_stats.items():
        if 'mem_after' in stat: # 确保 post_hook 被触发了
            mem_diff = stat['mem_after'] - base_mem
            print(f"{name:<45} | {stat['type']:<20} | {mem_diff:>10.2f} | {stat['time']:>8.4f}")
    
    for h in hooks:
        h.remove()
    
    torch.cuda.synchronize()
    total_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    print("-" * 110)
    print(f"Total Peak Memory: {total_mem:.2f} MB")
    print("=" * 110)

def run_detail_profiling():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("CUDA is not available. This script requires GPU for accurate memory/time profiling.")
        return

    # 设置相同的测试参数
    batch_size = 64
    in_channels = 64
    out_channels = 64
    h, w = 32, 60
    input_data = torch.randn(batch_size, in_channels, h, w).to(device)

    # 1. 分析 ResBlock
    res_block = ResBlock(in_channels, out_channels, stride=1).to(device)
    profile_layers("ResBlock (Baseline)", res_block, input_data)

    # 2. 分析 ECR (Normal Mode)
    ecr_block = EfficientCrossResBlock(in_channels, out_channels, num_evolve_layers=5, use_checkpoint=False).to(device)
    profile_layers("ECR L5 (Normal Mode)", ecr_block, input_data)

    # 3. 分析 ECR (Checkpoint Mode)
    ecr_block_cp = EfficientCrossResBlock(in_channels, out_channels, num_evolve_layers=5, use_checkpoint=True).to(device)
    profile_layers("ECR L5 (Checkpoint Mode)", ecr_block_cp, input_data)

if __name__ == "__main__":
    run_detail_profiling()
