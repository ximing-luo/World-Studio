"""
Sekiro-RL 模型统计工具 (Model Stats)
计算并分析模型的参数量、显存占用以及 FLOPs 估算。

使用方法:
python scripts/tools/stats.py
"""

import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import torch
import torch.nn as nn
import numpy as np
import torchvision.models as models

class MockSpace:
    def __init__(self, shape):
        self.shape = shape
        self.dtype = torch.uint8

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def analyze_memory(model, input_size, batch_size=64):
    # 模拟输入，使用与模型相同的设备
    device = next(model.parameters()).device
    x = torch.randn(batch_size, *input_size).to(device)
    
    print(f"\n{'='*100}")
    print(f"{'Layer (Type)':<30} | {'Input Shape':<20} | {'Output Shape':<20} | {'Params':<10} | {'Act. Mem (MB)':<8}")
    print(f"{'-'*100}")
    
    total_params = 0
    total_act_mem = 0
    total_flops = 0
    
    # 存储钩子句柄以便后续移除
    hooks = []
    
    # 仅针对叶子节点注册 Hook
    def register_hook(module):
        # 排除容器类（Sequential, ModuleList等），只关注实际计算层
        if len(list(module.children())) > 0:
            return

        def hook(m, input, output):
            nonlocal total_params, total_act_mem, total_flops
            
            class_name = str(m.__class__.__name__)
            
            # 获取输入形状
            if isinstance(input, tuple):
                input_shape = list(input[0].shape)
                inp = input[0]
            else:
                input_shape = list(input.shape)
                inp = input
            
            # 获取输出形状和元素数量
            if isinstance(output, (list, tuple)):
                output_shape = list(output[0].shape)
                out = output[0]
                num_elements = sum([o.numel() for o in output if isinstance(o, torch.Tensor)])
            else:
                output_shape = list(output.shape)
                out = output
                num_elements = output.numel()
            
            # 计算参数量（仅当前层）
            params = sum(p.numel() for p in m.parameters(recurse=False))
            total_params += params
            
            # 计算激活值显存 (FP32: 4 bytes)
            act_mem = num_elements * 4 / (1024**2)
            total_act_mem += act_mem

            # 估算 FLOPs (简单估算: Conv2d, Linear)
            flops = 0
            if isinstance(m, nn.Conv2d):
                # FLOPs = 2 * Cin * K * K * Hout * Wout * Cout
                # out.shape = [N, Cout, Hout, Wout]
                output_elements = out.numel()
                kernel_ops = m.in_channels * m.kernel_size[0] * m.kernel_size[1] // m.groups
                flops = 2 * output_elements * kernel_ops
            elif isinstance(m, nn.Linear):
                # FLOPs = 2 * Cin * Cout
                # out.shape = [N, Cout]
                flops = 2 * out.numel() * m.in_features
            
            total_flops += flops
            
            # 格式化输出
            in_shape_str = str(input_shape)
            out_shape_str = str(output_shape)
            
            # 截断过长的形状描述
            if len(in_shape_str) > 20: in_shape_str = "..." + in_shape_str[-17:]
            if len(out_shape_str) > 20: out_shape_str = "..." + out_shape_str[-17:]
            
            # print(f"{class_name:<30} | {in_shape_str:<20} | {out_shape_str:<20} | {params:<8,} | {act_mem:<12.2f}")

        hooks.append(module.register_forward_hook(hook))

    # 递归注册
    model.apply(register_hook)
    
    # 执行前向传播
    with torch.no_grad():
        model(x)
    
    # 移除钩子
    for h in hooks:
        h.remove()
        
    print(f"{'='*100}\n")
    
    # 汇总统计
    param_mem = total_params * 4 / (1024**2)        # FP32
    grad_mem = total_params * 4 / (1024**2)         # FP32
    optim_mem = total_params * 8 / (1024**2)        # Adam (m, v)
    
    # 计算算术强度 (Arithmetic Intensity)
    # Intensity = Total FLOPs / Total Memory Access (Bytes)
    # Memory Access = Read (Weights + Input Activations) + Write (Output Activations)
    # 简化估算: Weights + 2 * Activations (Read + Write)
    total_mem_bytes = (total_params * 4) + (total_act_mem * 1024**2 * 2) 
    arithmetic_intensity = total_flops / (total_mem_bytes + 1e-8)

    # 计算参数效率 (Parameter Efficiency)
    # 定义: 每消耗 1 GFLOPs 的算力，能利用多少 Million 参数
    # 公式: Params (M) / FLOPs_per_image (G)
    # 意义: 越高说明模型越"聪明且省力" (如 MoE, Transformer); 越低说明模型越"笨重且费力" (如 VGG)
    flops_per_image = total_flops / batch_size
    param_efficiency = (total_params / 1e6) / (flops_per_image / 1e9 + 1e-8)

    print(f"Model Summary (Batch Size: {batch_size})")
    print(f"{'-'*40}")
    print(f"Total Parameters:       {total_params / 1e6:.2f} M")
    print(f"Total FLOPs (Batch):    {total_flops / 1e9:.2f} GFLOPs")
    print(f"FLOPs per Image:        {flops_per_image / 1e9:.2f} GFLOPs")
    print(f"Arithmetic Intensity:   {arithmetic_intensity:.2f} OPS/Byte")
    print(f"Param Efficiency:       {param_efficiency:.2f} M_Params/GFLOPs")
    print(f"{'-'*40}")
    print(f"Weights Memory:         {param_mem:.2f} MB")
    print(f"Gradients Memory:       {grad_mem:.2f} MB")
    print(f"Optimizer Memory:       {optim_mem:.2f} MB (Adam)")
    print(f"Activations Memory:     {total_act_mem:.2f} MB (Forward pass)")
    print(f"{'-'*40}")
    print(f"Total Training Memory:  {param_mem + grad_mem + optim_mem + total_act_mem:.2f} MB")
    # 推理时通常 batch_size=1，激活值显存需按比例缩放
    inference_act_mem_b1 = total_act_mem / batch_size
    print(f"Inference Memory (B=1): {param_mem + inference_act_mem_b1:.2f} MB")
    print(f"{'='*40}\n")

if __name__ == "__main__":
    from src.model.sekiro.vae import ResNetSekiroVAE, EfficientCrossSekiroVAE
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 输入尺寸统一为 128x240 (Sekiro 项目标准)
    input_size = (3, 128, 240)
    batch_size = 1 # 推理视角对比
    
    print(f"\n{'#'*100}")
    print(f"{'VAE Architecture Comparison: ResNet vs. EfficientCross (ECR)':^100}")
    print(f"{'#'*100}\n")

    # 1. 分析传统的 ResNetSekiroVAE
    print(f"--- [Stage 1] Analyzing ResNetSekiroVAE (Standard Dense Residuals) ---")
    res_vae = ResNetSekiroVAE(latent_dim=256).to(device)
    analyze_memory(res_vae, input_size, batch_size=batch_size)
    
    # 2. 分析最新的 EfficientCrossSekiroVAE
    print(f"\n--- [Stage 2] Analyzing EfficientCrossSekiroVAE (Sparse Evolution + Cross Fusion) ---")
    ecr_vae = EfficientCrossSekiroVAE(latent_dim=256, num_hiddens=64).to(device)
    analyze_memory(ecr_vae, input_size, batch_size=batch_size)
