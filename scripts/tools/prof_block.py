import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.model.components.resnet import ResBlock
from src.model.components.rms import RMSNorm2d, RMSNorm
from src.model.components.attention import Focus, UnFocus
from src.model.components.ecr import (
    CrossScholarFusion, EfficientEvolutionLayer, EfficientCrossResBlock
)
from src.model.sekiro.vae import EfficientCrossSekiroVAE
from src.model.components.attention import ThinkingSpace
from src.model.backbone.transform import DeepSeekV3Block

def measure_latency(module, input_data, iterations=50):
    """
    使用 torch.cuda.Event 精确测量 GPU 耗时 (ms)
    """
    if not torch.cuda.is_available():
        return 0.0
    
    # Warmup - Increase for torch.compile
    for _ in range(20):
        with torch.no_grad():
            _ = module(input_data)
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(iterations):
        with torch.no_grad():
            _ = module(input_data)
    end_event.record()
    
    torch.cuda.synchronize()
    return start_event.elapsed_time(end_event) / iterations

def calculate_conv2d_flops(module, input_shape):
    # input_shape: (B, C, H, W)
    if not isinstance(module, nn.Conv2d):
        return 0, input_shape

    h_in, w_in = input_shape[2], input_shape[3]
    
    # Kernel size and stride might be int or tuple
    k_h = module.kernel_size[0] if isinstance(module.kernel_size, (list, tuple)) else module.kernel_size
    k_w = module.kernel_size[1] if isinstance(module.kernel_size, (list, tuple)) else module.kernel_size
    s_h = module.stride[0] if isinstance(module.stride, (list, tuple)) else module.stride
    s_w = module.stride[1] if isinstance(module.stride, (list, tuple)) else module.stride
    p_h = module.padding[0] if isinstance(module.padding, (list, tuple)) else module.padding
    p_w = module.padding[1] if isinstance(module.padding, (list, tuple)) else module.padding
    d_h = module.dilation[0] if isinstance(module.dilation, (list, tuple)) else module.dilation
    d_w = module.dilation[1] if isinstance(module.dilation, (list, tuple)) else module.dilation

    kernel_ops = k_h * k_w * (module.in_channels // module.groups)
    
    # Output shape calculation
    h_out = (h_in + 2 * p_h - d_h * (k_h - 1) - 1) // s_h + 1
    w_out = (w_in + 2 * p_w - d_w * (k_w - 1) - 1) // s_w + 1
    
    output_shape = (input_shape[0], module.out_channels, h_out, w_out)
    
    # 2 * MACs (Multiply-Add)
    flops = 2 * (h_out * w_out * module.out_channels * kernel_ops)
    return flops, output_shape

def calculate_evolution_flops(module, input_shape):
    # input_shape: (B, C, H, W)
    if isinstance(module, EfficientEvolutionLayer):
        # Evolution is 1x1 depthwise + 1x1 depthwise essentially
        f_conv = module.conv[0].in_channels * module.conv[0].kernel_size[0] * module.conv[0].kernel_size[1] * input_shape[2] * input_shape[3]
        return 2 * f_conv, input_shape
    return 0, input_shape

def calculate_mla_flops(module, input_shape):
    # MLA (Latent Attention) complexity
    # input_shape: (B, SeqLen, HiddenDim)
    b, s, d = input_shape
    # MLA involves:
    # 1. Latent Compression (d -> d_latent)
    # 2. Key/Value projection from latent
    # 3. Query projection (with latent)
    # 4. Attention (S*S*d_latent)
    # 5. Output projection
    # For simplified diagnosis, we assume MLA is ~4x standard MHA if ranks are similar
    # Here we use a conservative 2 * (S * d^2 + S^2 * d) baseline
    return 2 * (s * d * d + s * s * 32), input_shape

def calculate_moe_flops(module, input_shape):
    # SelfAdaptiveMoE complexity
    # input_shape: (B, SeqLen, HiddenDim)
    b, s, d = input_shape
    num_experts_per_tok = getattr(module, 'num_experts_per_tok', 2)
    intermediate_size = getattr(module, 'intermediate_size', d * 2)
    # Router: d * num_experts
    # Expert: d * intermediate_size * 2 (Linear x2)
    f_router = s * d * module.num_experts
    f_experts = s * num_experts_per_tok * (d * intermediate_size * 2)
    return 2 * (f_router + f_experts), input_shape

def calculate_deepseek_block_flops(module, input_shape):
    # input_shape: (B, SeqLen, HiddenDim)
    f_attn, _ = calculate_mla_flops(module.attn, input_shape)
    f_moe, _ = calculate_moe_flops(module.mlp, input_shape)
    return f_attn + f_moe, input_shape

def calculate_module_detailed(module, input_shape, return_sub_parts=False):
    """
    递归计算模块的 FLOPs 和输出形状。
    返回: (FLOPs, output_shape, output_mem_mb, total_train_mem_mb) 
          或 (FLOPs, output_shape, output_mem_mb, total_train_mem_mb, [sub_parts_info])
    """
    flops = 0
    out_shape = input_shape
    total_train_mem_mb = 0
    sub_parts_info = []
    
    if isinstance(module, nn.Sequential):
        for i, sub_module in enumerate(module):
            f, out_shape, m_out, m_train = calculate_module_detailed(sub_module, out_shape)
            flops += f
            total_train_mem_mb += m_train
            sub_parts_info.append((f"{module.__class__.__name__}.{i}", sub_module.__class__.__name__, out_shape, f, m_train))
    elif isinstance(module, Focus):
        # Focus is Space-to-Depth, no compute, just reshape
        b, c, h, w = input_shape
        out_shape = (b, c * (module.block_size**2), h // module.block_size, w // module.block_size)
    elif isinstance(module, UnFocus):
        # UnFocus is Depth-to-Space
        b, c, h, w = input_shape
        out_shape = (b, c // (module.block_size**2), h * module.block_size, w * module.block_size)
    elif isinstance(module, EfficientEvolutionLayer):
        flops, out_shape = calculate_evolution_flops(module, input_shape)
    elif isinstance(module, EfficientCrossResBlock):
        # 1. Expand
        f_exp, exp_out_shape, m_out_exp, m_train_exp = calculate_module_detailed(module.expand, out_shape)
        flops += f_exp
        total_train_mem_mb += m_train_exp
        sub_parts_info.append(("Expand", "nn.Sequential", exp_out_shape, f_exp, m_train_exp))
        
        # 2. Evolution
        f_evo, evo_out_shape, m_out_evo, m_train_evo = calculate_module_detailed(module.evolution, exp_out_shape)
        flops += f_evo
        
        if getattr(module, 'use_checkpoint', False):
            # Gradient Checkpointing: only store input activation + peak recompute activation
            # Simplified estimate: input + 1 layer peak
            evo_module = module.evolution
            # Handle compiled/scripted modules which might not have len()
            if hasattr(evo_module, "_orig_mod"): # torch.compile
                num_layers = len(evo_module._orig_mod)
            elif hasattr(evo_module, "graph"): # torch.jit.script
                # Rough estimate for scripted modules
                num_layers = module.num_evolve_layers if hasattr(module, "num_evolve_layers") else 18
            else:
                num_layers = len(evo_module)
            
            m_train_evo_optimized = m_out_exp + (m_train_evo / max(1, num_layers))
            total_train_mem_mb += m_train_evo_optimized
            sub_parts_info.append(("Evolution (Checkpoint)", "nn.Sequential", evo_out_shape, f_evo, m_train_evo_optimized))
        else:
            total_train_mem_mb += m_train_evo
            sub_parts_info.append(("Evolution", "nn.Sequential", evo_out_shape, f_evo, m_train_evo))
        
        # 3. Fusion (Cross Scholar)
        f_fus, fus_out_shape = calculate_conv2d_flops(nn.Conv2d(evo_out_shape[1], evo_out_shape[1], 1), evo_out_shape)
        # Fusion is complex, but its output is the main part
        num_elements = 1
        for dim in fus_out_shape:
            if dim != -1: num_elements *= dim
        m_train_fus = num_elements * 4 / (1024**2)
        flops += f_fus
        total_train_mem_mb += m_train_fus
        sub_parts_info.append(("Fusion", "CrossScholarFusion", fus_out_shape, f_fus, m_train_fus))
        out_shape = fus_out_shape
    elif isinstance(module, ResBlock):
        # 1. Residual Function
        f_res, res_out_shape, m_out_res, m_train_res = calculate_module_detailed(module.residual_function, out_shape)
        flops += f_res
        total_train_mem_mb += m_train_res
        sub_parts_info.append(("Residual Function", "nn.Sequential", res_out_shape, f_res, m_train_res))
        
        # 2. Res Norm
        f_norm, norm_out_shape, m_out_norm, m_train_norm = calculate_module_detailed(module.res_norm, res_out_shape)
        flops += f_norm
        total_train_mem_mb += m_train_norm
        sub_parts_info.append(("Res Norm", "RMSNorm2d", norm_out_shape, f_norm, m_train_norm))
        
        # 3. Shortcut
        f_short, short_out_shape, m_out_short, m_train_short = calculate_module_detailed(module.shortcut, input_shape)
        flops += f_short
        total_train_mem_mb += m_train_short
        sub_parts_info.append(("Shortcut", "nn.Sequential", short_out_shape, f_short, m_train_short))
        
        out_shape = norm_out_shape
        # Ensure we return the accumulated memory
        if return_sub_parts:
            return flops, out_shape, m_out_norm, total_train_mem_mb, sub_parts_info
        return flops, out_shape, m_out_norm, total_train_mem_mb
    elif isinstance(module, ThinkingSpace):
        b, c, h, w = input_shape
        current_seq_shape = (b, c, h * w) 
        for block in module.blocks:
            f_block, _ = calculate_deepseek_block_flops(block, current_seq_shape)
            flops += f_block
        out_shape = input_shape
    elif isinstance(module, nn.Conv2d):
        flops, out_shape = calculate_conv2d_flops(module, input_shape)
    elif isinstance(module, nn.Linear):
        flops = 2 * module.in_features * module.out_features * input_shape[0]
        out_shape = (input_shape[0], module.out_features)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.RMSNorm, RMSNorm2d, RMSNorm, nn.SiLU, nn.ReLU, nn.Sigmoid, nn.Flatten, nn.Upsample, nn.AvgPool2d, nn.Identity)):
        if isinstance(module, nn.Flatten):
            out_shape = (input_shape[0], -1)
        elif isinstance(module, nn.Upsample):
            if module.size is not None:
                out_shape = (input_shape[0], input_shape[1], module.size[0], module.size[1])
            else:
                out_shape = (input_shape[0], input_shape[1], int(input_shape[2] * module.scale_factor), int(input_shape[3] * module.scale_factor))
        elif isinstance(module, nn.AvgPool2d):
            # Simplified AvgPool output shape
            h_in, w_in = input_shape[2], input_shape[3]
            k = module.kernel_size if isinstance(module.kernel_size, int) else module.kernel_size[0]
            s = module.stride if isinstance(module.stride, int) else module.stride[0]
            p = module.padding if isinstance(module.padding, int) else module.padding[0]
            h_out = (h_in + 2 * p - k) // s + 1
            w_out = (w_in + 2 * p - k) // s + 1
            out_shape = (input_shape[0], input_shape[1], h_out, w_out)
        else:
            out_shape = input_shape

    # Calculate Output Activation Memory (MB)
    num_elements = 1
    for dim in out_shape:
        if dim != -1: num_elements *= dim
    output_mem_mb = num_elements * 4 / (1024**2)
    
    # If not a container, total_train_mem is just the output_mem
    if not isinstance(module, (nn.Sequential, EfficientCrossResBlock)):
        total_train_mem_mb = output_mem_mb
    
    if return_sub_parts:
        return flops, out_shape, output_mem_mb, total_train_mem_mb, sub_parts_info
    return flops, out_shape, output_mem_mb, total_train_mem_mb

def diagnose_ecr_block_internals():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*135}")
    print(f"{'EfficientCrossResBlock Internal Detailed Diagnosis (B=64, Optimized with Checkpoint)':^135}")
    print(f"{'='*135}")
    
    BATCH_SIZE = 64
    INPUT_SHAPE = (64, 64, 32, 60) # focus_out_channels = 64
    
    # Test with layers=18 to see memory growth
    NUM_EVOLVE_LAYERS = 18
    block = EfficientCrossResBlock(
        in_channels=64,
        num_evolve_layers=NUM_EVOLVE_LAYERS,
        expansion=1,
        stride=1,
        use_checkpoint=True
    ).to(device)
    block.train() # Checkpoint is only active in train mode
    
    # Pre-generate input
    dummy_input = torch.randn(BATCH_SIZE, 64, 32, 60).to(device)

    print(f"{'Component':<30} | {'Layer Type':<20} | {'Output Shape':<25} | {'FLOPs (M)':<12} | {'Train Mem (MB)':<15} | {'Latency (ms)':<12}")
    print("-" * 155)

    total_flops, final_shape, out_mem, total_train_mem, sub_parts = calculate_module_detailed(block, INPUT_SHAPE, return_sub_parts=True)
    
    # Measuring latencies
    with torch.no_grad():
        # 1. Expand
        t_exp = measure_latency(block.expand, dummy_input)
        
        # 2. Evolution
        exp_out = block.expand(dummy_input)
        t_evo = measure_latency(block.evolution, exp_out)
        
        # 3. Fusion
        evo_out = block.evolution(exp_out)
        t_fus = measure_latency(block.fusion, evo_out)
        
        # Total
        t_total = measure_latency(block, dummy_input)

    latencies = {"Expand": t_exp, "Evolution": t_evo, "Evolution (Checkpoint)": t_evo, "Fusion": t_fus}
    
    for name, l_type, shape, f, m_train in sub_parts:
        lat = latencies.get(name, 0.0)
        print(f"{name:<30} | {l_type:<20} | {str(shape):<25} | {f/1e6:<12.2f} | {m_train:<15.2f} | {lat:<12.4f}")
        
        # If it's Evolution, trace its sub-layers
        if name == "Evolution" or name == "Evolution (Checkpoint)":
            current_input = exp_out
            evo_layers = block.evolution
            if hasattr(evo_layers, "_orig_mod"):
                evo_layers = evo_layers._orig_mod
            elif hasattr(evo_layers, "graph"):
                # For scripted modules, we can't easily iterate layers
                print("  └─ (Optimized Sequential: Sub-layer tracing skipped)")
                continue
                
            for i, layer in enumerate(evo_layers):
                f_l, next_s, m_out_l, m_train_l = calculate_module_detailed(layer, shape)
                t_l = measure_latency(layer, current_input, iterations=20)
                if i < 2 or i >= len(evo_layers) - 1:
                    print(f"  └─ Layer_{i:<24} | {layer.__class__.__name__:<20} | {str(next_s):<25} | {f_l/1e6:<12.2f} | {m_train_l:<15.2f} | {t_l:<12.4f}")
                elif i == 2:
                    print(f"  └─ ... (hidden {len(evo_layers)-3} layers) ...")
                with torch.no_grad():
                    current_input = layer(current_input)

    print("-" * 155)
    print(f"EfficientCrossResBlock Total FLOPs: {total_flops/1e6:.2f} MFLOPs")
    print(f"EfficientCrossResBlock Measured Total Latency: {t_total:.4f} ms")
    # TFLOPS = FLOPs / (Time * 1e12)
    tflops = (total_flops / (t_total / 1000)) / 1e12 if t_total > 0 else 0
    print(f"EfficientCrossResBlock Hardware Utilization: {tflops:.4f} TFLOPS")
    print(f"EfficientCrossResBlock Total Training Act Mem (Accumulated): {total_train_mem:.2f} MB")
    
    total_params = sum(p.numel() for p in block.parameters())
    print(f"Block Params: {total_params/1e3:.2f} K")
    print(f"{'='*135}\n")

def diagnose_block_comparison():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*135}")
    print(f"{'Block Architecture Comparison: EfficientCrossResBlock vs ResBlock (B=64)':^135}")
    print(f"{'='*135}")
    
    INPUT_SHAPE = (64, 64, 32, 60)
    CHANNELS = 64
    
    # 1. EfficientCrossResBlock (18 layers)
    ecr_block = EfficientCrossResBlock(in_channels=CHANNELS, num_evolve_layers=18, expansion=1, stride=1).to(device)
    
    # 2. EfficientCrossResBlock (18 layers, with Checkpoint)
    ecr_opt_block = EfficientCrossResBlock(in_channels=CHANNELS, num_evolve_layers=18, expansion=1, stride=1, use_checkpoint=True).to(device)
    ecr_opt_block.train() # Must be in train mode for checkpointing
    
    # 3. ResBlock (Standard Inverted Bottleneck)
    res_block = ResBlock(in_channels=CHANNELS, out_channels=CHANNELS, stride=1).to(device)
    
    dummy_input = torch.randn(64, CHANNELS, 32, 60).to(device)

    print(f"{'Block Type':<30} | {'Params (K)':<12} | {'FLOPs (M)':<12} | {'Train Mem (MB)':<15} | {'Latency (ms)':<12} | {'Utilization'}")
    print("-" * 155)
    
    # Analyze ECR
    f_ecr, _, _, m_ecr = calculate_module_detailed(ecr_block, INPUT_SHAPE)
    p_ecr = sum(p.numel() for p in ecr_block.parameters())
    t_ecr = measure_latency(ecr_block, dummy_input)
    util_ecr = (f_ecr / (t_ecr / 1000)) / 1e12 if t_ecr > 0 else 0
    print(f"{'ECR Block (L18)':<30} | {p_ecr/1e3:<12.2f} | {f_ecr/1e6:<12.2f} | {m_ecr:<15.2f} | {t_ecr:<12.4f} | {util_ecr:.4f} TFLOPS")
    
    # Analyze ECR Optimized
    f_ecr_opt, _, _, m_ecr_opt = calculate_module_detailed(ecr_opt_block, INPUT_SHAPE)
    p_ecr_opt = p_ecr
    t_ecr_opt = measure_latency(ecr_opt_block, dummy_input)
    util_ecr_opt = (f_ecr_opt / (t_ecr_opt / 1000)) / 1e12 if t_ecr_opt > 0 else 0
    print(f"{'ECR Block (L18, Optimized)':<30} | {p_ecr_opt/1e3:<12.2f} | {f_ecr_opt/1e6:<12.2f} | {m_ecr_opt:<15.2f} | {t_ecr_opt:<12.4f} | {util_ecr_opt:.4f} TFLOPS")
    
    # Analyze ResBlock
    f_res, _, _, m_res, sub_res = calculate_module_detailed(res_block, INPUT_SHAPE, return_sub_parts=True)
    p_res = sum(p.numel() for p in res_block.parameters())
    t_res = measure_latency(res_block, dummy_input)
    util_res = (f_res / (t_res / 1000)) / 1e12 if t_res > 0 else 0
    
    print(f"{'ResBlock (Inverted Bottleneck)':<30} | {p_res/1e3:<12.2f} | {f_res/1e6:<12.2f} | {m_res:<15.2f} | {t_res:<12.4f} | {util_res:.4f} TFLOPS")
    
    print(f"\nResBlock Internal Breakdown:")
    for name, l_type, shape, f, m_train in sub_res:
        print(f"  {name:<30} | {l_type:<20} | {str(shape):<25} | {f/1e6:<12.2f} | {m_train:<15.2f}")
    
    print("-" * 155)
    print(f"Summary:")
    print(f"- FLOPs Reduction: {(1 - f_ecr/f_res)*100:.1f}%")
    print(f"- Params Reduction: {(1 - p_ecr/p_res)*100:.1f}%")
    print(f"- Memory Optimization (ECR-Opt vs Res): {(1 - m_ecr_opt/m_res)*100:.1f}% Reduction")
    print(f"- Speed Deficit: {t_ecr/t_res:.1f}x Slower (Due to Hardware Utilization Gap)")
    print(f"{'='*155}\n")

if __name__ == "__main__":
    diagnose_ecr_block_internals()
    diagnose_block_comparison()
