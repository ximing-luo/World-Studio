import torch
import torch.nn.functional as F
import os
import time
# 统一从 ops_evolution 导入加载好的 evolution_cuda 模块
from ops_evolution import evolution_cuda

def pytorch_evolution_forward(input_tensor, weights, biases):
    """仅保留分组卷积，移除 ReLU 和 Residual，测试 PyTorch 分组卷积的极限性能"""
    w = weights.unsqueeze(1)
    return F.conv2d(input_tensor, w, bias=biases, padding=1, groups=input_tensor.size(1))

def benchmark_speed():
    # 准备大型测试数据以模拟真实工作负载
    B, C, H, W = 64, 128, 64, 64
    input_tensor = torch.randn(B, C, H, W).cuda()
    weights = torch.randn(C, 3, 3).cuda() * 0.02
    biases = torch.randn(C).cuda() * 0.01

    print(f"\n[Speed Benchmark Config] B={B}, C={C}, H={H}, W={W}")
    print("-" * 50)

    warmup_iters = 50
    test_iters = 500

    # 预热 (Warmup)
    print(f"正在预热 ({warmup_iters} 次)...")
    for _ in range(warmup_iters):
        evolution_cuda.forward(input_tensor, weights, biases)
        pytorch_evolution_forward(input_tensor, weights, biases)
    torch.cuda.synchronize()

    # 基准测试工具
    def run_test(func, label):
        # 强制同步确保开始时间准确
        torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(test_iters):
            _ = func()
        # 强制同步确保结束时间准确
        torch.cuda.synchronize()
        return (time.time() - start_time) * 1000 / test_iters

    # 执行各版本测试
    cuda_time = run_test(lambda: evolution_cuda.forward(input_tensor, weights, biases), "CUDA 内核")
    torch_time = run_test(lambda: pytorch_evolution_forward(input_tensor, weights, biases), "PyTorch 深度卷积 (仅 Conv)")
    
    # 3x3 卷积对齐 (纯卷积)
    conv3x3_w = torch.randn(C, C, 3, 3).cuda()
    conv3x3_b = torch.randn(C).cuda()
    def standard_3x3_pure():
        return F.conv2d(input_tensor, conv3x3_w, bias=conv3x3_b, padding=1)
    conv3x3_time = run_test(standard_3x3_pure, "PyTorch 标准 3x3 (纯卷积)")
    
    # 1x1 卷积对齐 (纯卷积)
    conv1x1_w = torch.randn(C, C, 1, 1).cuda()
    conv1x1_b = torch.randn(C).cuda()
    def standard_1x1_pure():
        return F.conv2d(input_tensor, conv1x1_w, bias=conv1x1_b, padding=0)
    conv1x1_time = run_test(standard_1x1_pure, "PyTorch 标准 1x1 (纯卷积)")

    # 输出报告
    def calculate_flops(b, c, h, w, mode="dw"):
        base = b * c * h * w
        if mode == "dw":
            return base * (3 * 3 * 2 + 1 + 1 + 1) # 18(conv) + 1(relu) + 1(bias) + 1(res)
        elif mode == "3x3":
            return base * (c * 3 * 3 * 2 + 1 + 1 + 1)
        elif mode == "1x1":
            return base * (c * 1 * 1 * 2 + 1 + 1 + 1)
        return 0

    dw_flops = calculate_flops(B, C, H, W, "dw")
    conv3x3_flops = calculate_flops(B, C, H, W, "3x3")
    conv1x1_flops = calculate_flops(B, C, H, W, "1x1")

    def get_gflops(flops, time_ms):
        return (flops / 1e9) / (time_ms / 1000.0)

    print(f"\n[性能对比报告 (全流程对齐)]")
    print(f"{'实现方式':<30} | {'平均耗时 (ms)':<15} | {'算力利用 (GFLOPS)':<18} | {'相对加速比':<12}")
    print("-" * 90)
    print(f"{'PyTorch 深度卷积':<30} | {torch_time:>13.4f} ms | {get_gflops(dw_flops, torch_time):>16.2f} | {'1.00x':>10}")
    print(f"{'CUDA 内核 (含 ReLU+Res)':<30} | {cuda_time:>13.4f} ms | {get_gflops(dw_flops, cuda_time):>16.2f} | {torch_time / cuda_time:>9.2f}x")
    print(f"{'PyTorch 标准 3x3 (仅 Conv)':<30} | {conv3x3_time:>13.4f} ms | {get_gflops(conv3x3_flops, conv3x3_time):>16.2f} | {torch_time / conv3x3_time:>9.2f}x")
    print(f"{'PyTorch 标准 1x1 (仅 Conv)':<30} | {conv1x1_time:>13.4f} ms | {get_gflops(conv1x1_flops, conv1x1_time):>16.2f} | {torch_time / conv1x1_time:>9.2f}x")
    print("-" * 90)

if __name__ == "__main__":
    benchmark_speed()
