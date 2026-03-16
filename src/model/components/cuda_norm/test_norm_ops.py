import torch
import torch.nn.functional as F
import time
import sys
import os

# 将项目根目录添加到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))

from src.model.components.norm import RMSNorm2d, LayerNorm2d

def get_peak_memory():
    # 返回以 MB 为单位的峰值显存占用
    mem = torch.cuda.max_memory_allocated() / (1024 * 1024)
    torch.cuda.reset_peak_memory_stats()
    return mem

def test_norm_correctness():
    print(">>> Testing Correctness...")
    device = torch.device("cuda")
    B, C, H, W = 16, 256, 16, 16
    x = torch.randn(B, C, H, W, device=device, requires_grad=True)
    
    # 1. Test RMSNorm2d
    rms_op = RMSNorm2d(C).to(device)
    
    # 原生 Python 实现 (用于对比)
    def native_rms(x, weight, eps=1e-4):
        return x * torch.rsqrt(x.float().pow(2).mean(1, keepdim=True) + eps).type_as(x) * weight

    out_cuda = rms_op(x)
    out_native = native_rms(x, rms_op.weight)
    
    diff = (out_cuda - out_native).abs().max().item()
    print(f"RMSNorm2d Fwd Diff: {diff:.8f}")
    
    # Test Backward
    out_cuda.backward(torch.ones_like(out_cuda))
    grad_cuda = x.grad.clone()
    x.grad.zero_()
    
    out_native.backward(torch.ones_like(out_native))
    grad_native = x.grad.clone()
    
    grad_diff = (grad_cuda - grad_native).abs().max().item()
    print(f"RMSNorm2d Bwd Diff: {grad_diff:.8f}")

    x_ln = torch.randn(B, C, H, W, device=device, requires_grad=True)
    ln_op = LayerNorm2d(C).to(device)

    def native_ln(x, weight, bias, eps=1e-6):
        mean = x.mean(1, keepdim=True)
        var = (x - mean).pow(2).mean(1, keepdim=True)
        return (x - mean) * torch.rsqrt(var + eps) * weight + bias

    out_ln_cuda = ln_op(x_ln)
    out_ln_native = native_ln(x_ln, ln_op.weight, ln_op.bias)
    ln_fwd_diff = (out_ln_cuda - out_ln_native).abs().max().item()
    print(f"LayerNorm2d Fwd Diff: {ln_fwd_diff:.8f}")

    out_ln_cuda.backward(torch.ones_like(out_ln_cuda))
    grad_ln_cuda = x_ln.grad.clone()
    x_ln.grad.zero_()
    ln_w_grad_cuda = ln_op.weight.grad.clone()
    ln_b_grad_cuda = ln_op.bias.grad.clone()
    ln_op.weight.grad.zero_()
    ln_op.bias.grad.zero_()

    out_ln_native.backward(torch.ones_like(out_ln_native))
    grad_ln_native = x_ln.grad.clone()
    ln_w_grad_native = ln_op.weight.grad.clone()
    ln_b_grad_native = ln_op.bias.grad.clone()

    ln_bwd_diff = (grad_ln_cuda - grad_ln_native).abs().max().item()
    ln_w_diff = (ln_w_grad_cuda - ln_w_grad_native).abs().max().item()
    ln_b_diff = (ln_b_grad_cuda - ln_b_grad_native).abs().max().item()
    print(f"LayerNorm2d Bwd Diff: {ln_bwd_diff:.8f}")
    print(f"LayerNorm2d dW Diff: {ln_w_diff:.8f}")
    print(f"LayerNorm2d dB Diff: {ln_b_diff:.8f}")

def test_performance():
    print("\n>>> Testing Performance (B=64, C=1024, H=16, W=16)...")
    device = torch.device("cuda")
    B, C, H, W = 64, 1024, 16, 16
    x = torch.randn(B, C, H, W, device=device)
    
    # RMSNorm
    rms_op = RMSNorm2d(C).to(device)
    weight_rms = rms_op.weight.view(-1)
    
    # --- 1. Native Python (Multiple Operators) ---
    def native_rms_py(x, weight, eps=1e-4):
        return x * torch.rsqrt(x.float().pow(2).mean(1, keepdim=True) + eps).type_as(x) * weight
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        _ = native_rms_py(x, rms_op.weight)
    torch.cuda.synchronize()
    py_rms_time = (time.time() - start) * 10
    py_rms_mem = get_peak_memory()
    print(f"Native RMS (Python) | Time: {py_rms_time:.4f} ms | Peak Mem: {py_rms_mem:.2f} MB")

    # --- 2. Permute + Native (Torch C++ Optimized) ---
    def native_rms_permute(x, weight, eps=1e-4):
        # x: (B, C, H, W) -> (B, H, W, C)
        y = x.permute(0, 2, 3, 1).contiguous()
        # 使用官方 F.rms_norm (Torch 2.4+)，它是高度优化的 1D CUDA 核
        # 这也是为了公平对比：官方 1D 优化版 + Permute vs 我们的 2D 融合版
        res = F.rms_norm(y, (C,), weight, eps)
        return res.permute(0, 3, 1, 2).contiguous()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        _ = native_rms_permute(x, weight_rms)
    torch.cuda.synchronize()
    perm_rms_time = (time.time() - start) * 10
    perm_rms_mem = get_peak_memory()
    print(f"Native RMS (Permute)| Time: {perm_rms_time:.4f} ms | Peak Mem: {perm_rms_mem:.2f} MB")

    # --- 2.5 Pure 1D Native (No Layout Overhead) ---
    # 模拟输入本来就是 (N, C) 布局的情况，看看 1D 核的最快表现
    x_1d = x.permute(0, 2, 3, 1).reshape(-1, C).contiguous()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        _ = F.rms_norm(x_1d, (C,), weight_rms, 1e-4)
    torch.cuda.synchronize()
    pure_1d_rms_time = (time.time() - start) * 10
    pure_1d_rms_mem = get_peak_memory()
    print(f"Native RMS (Pure 1D)| Time: {pure_1d_rms_time:.4f} ms | Peak Mem: {pure_1d_rms_mem:.2f} MB")

    # --- 3. CUDA Fused (Our Coalesced Version) ---
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        _ = rms_op(x)
    torch.cuda.synchronize()
    fused_rms_time = (time.time() - start) * 10
    fused_rms_mem = get_peak_memory()
    print(f"CUDA Fused RMS      | Time: {fused_rms_time:.4f} ms | Peak Mem: {fused_rms_mem:.2f} MB")

    # LayerNorm
    ln_op = LayerNorm2d(C).to(device)
    weight_ln = ln_op.weight.view(-1)
    bias_ln = ln_op.bias.view(-1)

    # --- 1. Native Python ---
    def native_ln_py(x, weight, bias, eps=1e-6):
        mean = x.mean(1, keepdim=True)
        var = (x - mean).pow(2).mean(1, keepdim=True)
        return (x - mean) * torch.rsqrt(var + eps) * weight + bias

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        _ = native_ln_py(x, ln_op.weight, ln_op.bias)
    torch.cuda.synchronize()
    py_ln_time = (time.time() - start) * 10
    py_ln_mem = get_peak_memory()
    print(f"\nNative LN (Python)  | Time: {py_ln_time:.4f} ms | Peak Mem: {py_ln_mem:.2f} MB")

    # --- 2. Permute + Native (F.layer_norm) ---
    def native_ln_permute(x, weight, bias, eps=1e-6):
        # x: (B, C, H, W) -> (B, H, W, C)
        y = x.permute(0, 2, 3, 1).contiguous()
        res = F.layer_norm(y, (C,), weight, bias, eps)
        return res.permute(0, 3, 1, 2).contiguous()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        _ = native_ln_permute(x, weight_ln, bias_ln)
    torch.cuda.synchronize()
    perm_ln_time = (time.time() - start) * 10
    perm_ln_mem = get_peak_memory()
    print(f"Native LN (Permute) | Time: {perm_ln_time:.4f} ms | Peak Mem: {perm_ln_mem:.2f} MB")

    # --- 2.5 Pure 1D Native (No Layout Overhead) ---
    x_1d_ln = x.permute(0, 2, 3, 1).reshape(-1, C).contiguous()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        _ = F.layer_norm(x_1d_ln, (C,), weight_ln, bias_ln, 1e-6)
    torch.cuda.synchronize()
    pure_1d_ln_time = (time.time() - start) * 10
    pure_1d_ln_mem = get_peak_memory()
    print(f"Native LN (Pure 1D) | Time: {pure_1d_ln_time:.4f} ms | Peak Mem: {pure_1d_ln_mem:.2f} MB")

    # --- 3. CUDA Fused (Our Coalesced Version) ---
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        _ = ln_op(x)
    torch.cuda.synchronize()
    fused_ln_time = (time.time() - start) * 10
    fused_ln_mem = get_peak_memory()
    print(f"CUDA Fused LN       | Time: {fused_ln_time:.4f} ms | Peak Mem: {fused_ln_mem:.2f} MB")

if __name__ == "__main__":
    test_norm_correctness()
    test_performance()
