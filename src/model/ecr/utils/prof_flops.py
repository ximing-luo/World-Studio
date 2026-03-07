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
from .prof_utils import ManualFlopCounter

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
