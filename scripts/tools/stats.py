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
    # 模拟输入
    x = torch.randn(batch_size, *input_size)
    
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
            
            print(f"{class_name:<30} | {in_shape_str:<20} | {out_shape_str:<20} | {params:<8,} | {act_mem:<12.2f}")

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
    input_shape = (3, 136, 240)
    mock_space = MockSpace(input_shape)

    # print("\n" + "="*40 + " Original Model " + "="*40)
    # try:
    #     extractor = SekiroStableExtractor(mock_space)
    #     analyze_memory(extractor, input_shape, batch_size=32)
    # except Exception as e:
    #     print(f"Error analyzing Original Model: {e}")

    # print("\n" + "="*40 + " ResNet-like Model " + "="*40)
    # try:
    #     extractor_resnet = ResNetSpatial(mock_space)
    #     analyze_memory(extractor_resnet, input_shape, batch_size=32)
    # except Exception as e:
    #     print(f"Error analyzing ResNet-like Model: {e}")

    # print("\n" + "="*40 + " MobileNetV2-like Model " + "="*40)
    # try:
    #     extractor_mobilenet = MobileNetSpatial(mock_space)
    #     analyze_memory(extractor_mobilenet, input_shape, batch_size=32)
    # except Exception as e:
    #     print(f"Error analyzing MobileNetV2-like Model: {e}")

    # print("\n" + "="*40 + " EfficientNet-B0-like Model " + "="*40)
    # try:
    #     extractor_efficientnet = EfficientNetSpatial(mock_space)
    #     analyze_memory(extractor_efficientnet, input_shape, batch_size=32)
    # except Exception as e:
    #     print(f"Error analyzing EfficientNet-B0-like Model: {e}")

    print("\n\n" + "#"*40 + " Torchvision Models (No Weights) " + "#"*40)

    # 1. ResNet50
    print("\n" + "="*40 + " Torchvision ResNet50 (Modified Head) " + "="*40)
    try:
        # weights=None 表示不下载预训练权重 (随机初始化)
        resnet50 = models.resnet50(weights=None)
        # 修改全连接层以输出 512 维特征 (原始是 1000)
        resnet50.fc = nn.Linear(resnet50.fc.in_features, 512)
        analyze_memory(resnet50, input_shape, batch_size=32)
    except Exception as e:
        print(f"Error analyzing Torchvision ResNet50: {e}")

    # 2. MobileNetV2
    print("\n" + "="*40 + " Torchvision MobileNetV2 (Modified Head) " + "="*40)
    try:
        mobilenet_v2 = models.mobilenet_v2(weights=None)
        # 修改分类器最后一层
        # classifier 是一个 Sequential:
        # (0): Dropout(p=0.2, inplace=False)
        # (1): Linear(in_features=1280, out_features=1000, bias=True)
        mobilenet_v2.classifier[1] = nn.Linear(mobilenet_v2.last_channel, 512)
        analyze_memory(mobilenet_v2, input_shape, batch_size=32)
    except Exception as e:
        print(f"Error analyzing Torchvision MobileNetV2: {e}")

    # 3. EfficientNet-B0
    print("\n" + "="*40 + " Torchvision EfficientNet-B0 (Modified Head) " + "="*40)
    try:
        efficientnet_b0 = models.efficientnet_b0(weights=None)
        # 修改分类器最后一层
        # classifier 是一个 Sequential:
        # (0): Dropout(p=0.2, inplace=True)
        # (1): Linear(in_features=1280, out_features=1000, bias=True)
        efficientnet_b0.classifier[1] = nn.Linear(efficientnet_b0.classifier[1].in_features, 512)
        analyze_memory(efficientnet_b0, input_shape, batch_size=32)
    except Exception as e:
        print(f"Error analyzing Torchvision EfficientNet-B0: {e}")
