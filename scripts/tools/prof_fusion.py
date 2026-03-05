import torch
import torch.nn as nn

def calculate_conv2d_flops(c_in, c_out, kernel_size=1, groups=1, h=64, w=64):
    """
    计算 Conv2d 的 FLOPs (2 * MACs)
    """
    # 每组内的计算量 = (输入通道/组数) * (输出通道/组数) * 卷积核大小 * 特征图大小
    # 总计算量 = 每组计算量 * 组数
    # 化简后: (c_in * c_out / groups) * kernel_size^2 * h * w
    macs = (c_in * c_out // groups) * (kernel_size**2) * h * w
    flops = 2 * macs
    return flops

def analyze_fusion_strategies():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running Analysis on {device}\n")
    
    H, W = 64, 64
    C_BASE = 128
    C_MID = 512
    print(f"Input Shape: (B, {C_MID}, {H}, {W}) [Mid-Channels case]\n")
    
    # --- 策略 A: 标准 Dense 1x1 ---
    print("="*60)
    print(f"Strategy A: Standard Dense 1x1 Fusion ({C_MID} -> {C_MID})")
    print("="*60)
    flops_a = calculate_conv2d_flops(C_MID, C_MID, groups=1, h=H, w=W)
    print(f"Layer: Conv2d({C_MID}, {C_MID}, k=1, groups=1) | FLOPs: {flops_a/1e6:.2f} M")
    print(f"Total FLOPs (A): {flops_a/1e6:.2f} M")
    print("-" * 60)
    
    # --- 策略 B: 漏斗架构 (Funnel Fusion) ---
    print("\n" + "="*60)
    print("Strategy B: Funnel Fusion (Shrink128 -> Shrink32 -> Think -> Expand)")
    print("="*60)
    # 512 -> 128 (G128)
    f1 = calculate_conv2d_flops(C_MID, 128, groups=128, h=H, w=W)
    print(f"Step 1 (G128): Conv2d({C_MID}, 128, k=1, groups=128) | FLOPs: {f1/1e6:.2f} M")
    # 128 -> 32 (G32)
    f2 = calculate_conv2d_flops(128, 32, groups=32, h=H, w=W)
    print(f"Step 2 (G32):  Conv2d(128, 32, k=1, groups=32)  | FLOPs: {f2/1e6:.2f} M")
    # 32 -> 128 (Dense)
    f3 = calculate_conv2d_flops(32, 128, groups=1, h=H, w=W)
    print(f"Step 3 (Think): Conv2d(32, 128, k=1, groups=1)  | FLOPs: {f3/1e6:.2f} M")
    # 128 -> 512 (G128)
    f4 = calculate_conv2d_flops(128, C_MID, groups=128, h=H, w=W)
    print(f"Step 4 (Expand):Conv2d(128, {C_MID}, k=1, groups=128)| FLOPs: {f4/1e6:.2f} M")
    
    total_b = f1 + f2 + f3 + f4
    print("-" * 60)
    print(f"Total FLOPs (B): {total_b/1e6:.2f} M")
    
    # --- 策略 C: 静态学者融合 (Static Scholar Fusion) ---
    print("\n" + "="*60)
    print(f"Strategy C: Static Scholar Fusion ({C_MID} -> 16 -> {C_MID})")
    print("="*60)
    # 1. 静态归类 (512 -> 16, Dense)
    f_c1 = calculate_conv2d_flops(C_MID, 16, groups=1, h=H, w=W)
    print(f"Step 1 (Classify): Conv2d({C_MID}, 16, k=1, groups=1) | FLOPs: {f_c1/1e6:.2f} M")
    # 2. 个性化检索 (16 -> 512, Dense)
    f_c2 = calculate_conv2d_flops(16, C_MID, groups=1, h=H, w=W)
    print(f"Step 2 (Retrieve): Conv2d(16, {C_MID}, k=1, groups=1) | FLOPs: {f_c2/1e6:.2f} M")
    
    total_c = f_c1 + f_c2
    print("-" * 60)
    print(f"Total FLOPs (C): {total_c/1e6:.2f} M")
    
    # --- 对比 ---
    print("\n" + "="*60)
    print("FINAL COMPARISON (Mid-Channels: 512)")
    print("="*60)
    print(f"Strategy A (Dense 1x1):    {flops_a/1e6:>10.2f} M")
    print(f"Strategy B (Funnel):       {total_b/1e6:>10.2f} M")
    print(f"Strategy C (Scholar 16):   {total_c/1e6:>10.2f} M")
    print("-" * 60)
    print(f"Gain (A vs C): {flops_a/total_c:.2f}x Faster!")
    print(f"Efficiency: C is {total_b/total_c:.2f}x the compute of B (but no random grouping)")
    print("="*60)

if __name__ == "__main__":
    analyze_fusion_strategies()
