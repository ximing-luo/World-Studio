import torch
import torch.nn as nn
import sys
import os
import time
from collections import OrderedDict

# 添加项目根目录到路径
root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(root_path)

from src.model.components.resnet import BottleNeck, BasicBlock, TraditionalBasicBlock
from src.model.ecr.ecr import EfficientCrossResBlock, CrossScholarFusion
from src.model.components.attention import SEBlock
from src.model.ecr.cuda_evolution.ops_evolution import EvolutionLayer
from src.model.components.norm import RMSNorm2d, LayerNorm2d

def profile_layers(name, model, input_data):
    print(f"\n>>> Profiling: {name}")
    print("=" * 130)
    
    device = input_data.device
    model.to(device)
    model.train()
    
    # ---------------------------------------------------------
    # 关键修复：提前禁用所有层的 inplace 操作
    # 1. 避开反向传播 Hook 的限制
    # 2. 防止 input_data (leaf variable with requires_grad=True) 被第一层 inplace 修改报错
    # ---------------------------------------------------------
    original_inplace = {}
    for n, m in model.named_modules():
        if hasattr(m, 'inplace'):
            original_inplace[n] = m.inplace
            m.inplace = False
    
    hooks = []
    try:
        # ---------------------------------------------------------
        # 第一阶段: 纯净总耗时测试 (无钩子, 无同步, 无克隆)
        # ---------------------------------------------------------
        # 深度预热：包含前向和反向
        for _ in range(10):
            model.zero_grad(set_to_none=True)
            output = model(input_data)
            loss = output.sum()
            loss.backward()
        
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        
        clean_fwd_start = torch.cuda.Event(enable_timing=True)
        clean_fwd_end = torch.cuda.Event(enable_timing=True)
        clean_bwd_start = torch.cuda.Event(enable_timing=True)
        clean_bwd_end = torch.cuda.Event(enable_timing=True)
        
        torch.cuda.synchronize()
        clean_fwd_start.record()
        output = model(input_data)
        clean_fwd_end.record()
        
        loss = output.sum()
        torch.cuda.synchronize()
        clean_bwd_start.record()
        loss.backward()
        clean_bwd_end.record()
        model.zero_grad(set_to_none=True)
        
        torch.cuda.synchronize()
        pure_fwd_time = clean_fwd_start.elapsed_time(clean_fwd_end)
        pure_bwd_time = clean_bwd_start.elapsed_time(clean_bwd_end)
        pure_step_time = clean_fwd_start.elapsed_time(clean_bwd_end)
        pure_peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

        # ---------------------------------------------------------
        # 第二阶段: 逐层详细分析 (带钩子, 强制同步, 克隆)
        # ---------------------------------------------------------
        layer_stats = OrderedDict()
        
        def get_pre_hook(name):
            def hook(module, input):
                torch.cuda.synchronize()
                start_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                mem_before = torch.cuda.memory_allocated(device) / (1024 ** 2)
                if name not in layer_stats:
                    layer_stats[name] = {'type': module.__class__.__name__}
                layer_stats[name].update({
                    'fwd_start': start_event,
                    'mem_before': mem_before
                })
            return hook

        def get_post_hook(name):
            def hook(module, input, output):
                torch.cuda.synchronize()
                end_event = torch.cuda.Event(enable_timing=True)
                end_event.record()
                mem_after = torch.cuda.memory_allocated(device) / (1024 ** 2)
                
                layer_stats[name].update({
                    'mem_after': mem_after,
                    'fwd_end': end_event
                })
                
                if isinstance(output, torch.Tensor):
                    return output.clone()
                return output
            return hook

        def get_bwd_pre_hook(name):
            def hook(module, grad_output):
                torch.cuda.synchronize()
                start_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                layer_stats[name]['bwd_start'] = start_event
                return None
            return hook

        def get_bwd_post_hook(name):
            def hook(module, grad_input, grad_output):
                torch.cuda.synchronize()
                end_event = torch.cuda.Event(enable_timing=True)
                end_event.record()
                layer_stats[name]['bwd_end'] = end_event
                return None
            return hook

        for n, m in model.named_modules():
            if isinstance(m, nn.Identity):
                continue
            hooks.append(m.register_forward_pre_hook(get_pre_hook(n)))
            hooks.append(m.register_forward_hook(get_post_hook(n)))
            if len(list(m.children())) == 0 or isinstance(m, (EvolutionLayer, RMSNorm2d, LayerNorm2d)):
                try:
                    hooks.append(m.register_full_backward_pre_hook(get_bwd_pre_hook(n)))
                    hooks.append(m.register_full_backward_hook(get_bwd_post_hook(n)))
                except Exception as e:
                    if n in layer_stats:
                        layer_stats[n]['bwd_error'] = str(e)

        for _ in range(5):
            model.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(True):
                output = model(input_data)
                loss = output.sum()
                loss.backward()
        
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        base_mem = torch.cuda.memory_allocated(device) / (1024 ** 2)
        
        fwd_start = torch.cuda.Event(enable_timing=True)
        fwd_end = torch.cuda.Event(enable_timing=True)
        bwd_start = torch.cuda.Event(enable_timing=True)
        bwd_end = torch.cuda.Event(enable_timing=True)
        
        torch.cuda.synchronize()
        fwd_start.record()
        with torch.set_grad_enabled(True):
            output = model(input_data)
            fwd_end.record()
            loss = output.sum()
            torch.cuda.synchronize()
            bwd_start.record()
            loss.backward()
            bwd_end.record()
            
        torch.cuda.synchronize()
        total_fwd_time = fwd_start.elapsed_time(fwd_end)
        total_bwd_time = bwd_start.elapsed_time(bwd_end)
        total_step_time = fwd_start.elapsed_time(bwd_end)

        # ---------------------------------------------------------
        # 打印结果
        # ---------------------------------------------------------
        print(f"[Phase 1: Pure Performance (No Hooks)]")
        print(f"Total Forward Time:      {pure_fwd_time:>10.4f} ms")
        print(f"Total Backward Time:     {pure_bwd_time:>10.4f} ms")
        print(f"Total Training Step:     {pure_step_time:>10.4f} ms")
        print(f"Total Peak Memory:       {pure_peak_mem:>10.2f} MB")
        print("-" * 135)
        
        print(f"[Phase 2: Detailed Layer Breakdown (With Profiling Overhead)]")
        print(f"{'Layer Name':<25} | {'Type':<18} | {'Inc (MB)':<10} | {'Total (MB)':<10} | {'Fwd (ms)':<10} | {'Bwd (ms)':<10}")
        print("-" * 135)

        prev_mem = base_mem
        for name, stat in layer_stats.items():
            m = dict(model.named_modules()).get(name)
            if m is None: continue
            is_leaf = len(list(m.children())) == 0
            is_custom = isinstance(m, (EvolutionLayer, RMSNorm2d, LayerNorm2d, CrossScholarFusion, SEBlock))
            
            if (is_leaf or is_custom) and 'mem_after' in stat:
                mem_inc = stat['mem_after'] - prev_mem
                total_acc = stat['mem_after'] - base_mem
                fwd_t = 0.0
                if 'fwd_start' in stat and 'fwd_end' in stat:
                    fwd_t = stat['fwd_start'].elapsed_time(stat['fwd_end'])
                bwd_t = 0.0
                if 'bwd_start' in stat and 'bwd_end' in stat:
                    bwd_t = stat['bwd_start'].elapsed_time(stat['bwd_end'])
                bwd_str = f"{bwd_t:>8.4f}"
                if bwd_t == 0 and 'bwd_error' in stat:
                    bwd_str = "  Error "
                print(f"{name:<25} | {stat['type']:<18} | {mem_inc:>10.2f} | {total_acc:>10.2f} | {fwd_t:>8.4f} | {bwd_str}")
                prev_mem = stat['mem_after']
    finally:
        for h in hooks:
            h.remove()
        for n, val in original_inplace.items():
            m = dict(model.named_modules()).get(n)
            if m: m.inplace = val
        torch.cuda.synchronize()

    print("=" * 130)

def run_detail_profiling():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("CUDA is not available. This script requires GPU for accurate memory/time profiling.")
        return

    # 设置相同的测试参数 (高通道低分辨率)
    batch_size = 64
    in_channels = 256
    out_channels = 256
    h, w = 16, 16
    input_data = torch.randn(batch_size, in_channels, h, w, device=device, requires_grad=True)

    # 1. 分析 BottleNeck (通道数不变)
    bottleneck_block = BottleNeck(in_channels, out_channels, stride=1).to(device)
    profile_layers("BottleNeck (Baseline)", bottleneck_block, input_data)

    # 1.1 分析 TraditionalBasicBlock (通道数不变)
    traditional_basic_block = TraditionalBasicBlock(in_channels, out_channels, stride=1).to(device)
    profile_layers("Traditional BasicBlock (Standard 3x3)", traditional_basic_block, input_data)

    # 1.5 分析 BasicBlock (通道数不变)
    basic_block = BasicBlock(in_channels, out_channels, stride=1).to(device)
    profile_layers("BasicBlock (ResNet-D/18/34)", basic_block, input_data)

    # 2. 分析 ECR (通道数不变, 默认 expansion=1, evolution_layers=4)
    ecr_block = EfficientCrossResBlock(in_channels, out_channels, stride=1).to(device)
    profile_layers("ECR Block (4-Layer Evolution)", ecr_block, input_data)

if __name__ == "__main__":
    run_detail_profiling()
