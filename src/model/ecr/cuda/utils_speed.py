import torch
import torch.nn.functional as F
import os
import time
from torch.utils.cpp_extension import load

# 获取当前脚本所在目录
cuda_path = os.path.dirname(os.path.abspath(__file__))

def pytorch_evolution_forward(input_tensor, weights, biases):
    """等效的 PyTorch 实现，用于性能基准对比 (含 ReLU + Residual)"""
    x = F.relu(input_tensor)
    w = weights.unsqueeze(1)
    conv_res = F.conv2d(x, w, bias=biases, padding=1, groups=input_tensor.size(1))
    return input_tensor + conv_res

def benchmark_speed():
    # 1. 编译并加载原内核
    print("正在加载原内核...")
    try:
        evolution_lib = load(
            name="evolution_lib",
            sources=[
                os.path.join(cuda_path, "evolution_v1_bind.cpp"),
                os.path.join(cuda_path, "evolution_v1_kernel.cu")
            ],
            verbose=False
        )
    except Exception as e:
        print(f"原内核编译加载失败: {e}")
        return

    # 2. 编译并加载优化后的内核 (PREF)
    print("正在加载优化后的内核 (PREF)...")
    try:
        evolution_lib_pref = load(
            name="evolution_lib_pref",
            sources=[
                os.path.join(cuda_path, "evolution_v1_bind.cpp"),
                os.path.join(cuda_path, "evolution_v1_kernel.cu")
            ],
            verbose=False
        )
    except Exception as e:
        print(f"优化内核编译加载失败: {e}")
        return

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
        evolution_lib.forward(input_tensor, weights, biases)
        evolution_lib_pref.forward(input_tensor, weights, biases)
        pytorch_evolution_forward(input_tensor, weights, biases)
    torch.cuda.synchronize()

    # 基准测试工具
    def run_test(func, label):
        print(f"正在测试 {label} ({test_iters} 次)...")
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(test_iters):
            func()
        end_event.record()
        torch.cuda.synchronize()
        return start_event.elapsed_time(end_event) / test_iters

    # 执行各版本测试
    cuda_time = run_test(lambda: evolution_lib.forward(input_tensor, weights, biases), "原 CUDA 内核 (ReLU+Res)")
    pref_cuda_time = run_test(lambda: evolution_lib_pref.forward(input_tensor, weights, biases), "优化 CUDA 内核 (PREF, ReLU+Res)")
    torch_time = run_test(lambda: pytorch_evolution_forward(input_tensor, weights, biases), "PyTorch 深度卷积 (ReLU+Res)")
    
    # 3x3 卷积对齐 (含 ReLU + Residual)
    conv3x3_w = torch.randn(C, C, 3, 3).cuda()
    conv3x3_b = torch.randn(C).cuda()
    def standard_3x3_with_extras():
        x = F.relu(input_tensor)
        conv_res = F.conv2d(x, conv3x3_w, bias=conv3x3_b, padding=1)
        return input_tensor + conv_res
    conv3x3_time = run_test(standard_3x3_with_extras, "PyTorch 标准 3x3 (ReLU+Res)")
    
    # 1x1 卷积对齐 (含 ReLU + Residual)
    conv1x1_w = torch.randn(C, C, 1, 1).cuda()
    conv1x1_b = torch.randn(C).cuda()
    def standard_1x1_with_extras():
        x = F.relu(input_tensor)
        conv_res = F.conv2d(x, conv1x1_w, bias=conv1x1_b, padding=0)
        return input_tensor + conv_res
    conv1x1_time = run_test(standard_1x1_with_extras, "PyTorch 标准 1x1 (ReLU+Res)")

    # 输出报告
    def calculate_flops(b, c, h, w, mode="dw"):
        # 计算总操作数 (Floating Point Operations)
        # mode "dw": Depthwise 3x3 + ReLU + Bias + Residual
        # mode "3x3": Standard 3x3 + ReLU + Bias + Residual
        # mode "1x1": Standard 1x1 + ReLU + Bias + Residual
        base = b * c * h * w
        if mode == "dw":
            return base * (3 * 3 * 2 + 1 + 1 + 1) # 18(conv) + 1(relu) + 1(bias) + 1(res)
        elif mode == "3x3":
            return base * (c * 3 * 3 * 2 + 1 + 1 + 1) # 18*C(conv) + 3
        elif mode == "1x1":
            return base * (c * 1 * 1 * 2 + 1 + 1 + 1) # 2*C(conv) + 3
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
    print(f"{'原 CUDA 内核':<30} | {cuda_time:>13.4f} ms | {get_gflops(dw_flops, cuda_time):>16.2f} | {torch_time / cuda_time:>9.2f}x")
    print(f"{'优化 CUDA 内核 (PREF)':<30} | {pref_cuda_time:>13.4f} ms | {get_gflops(dw_flops, pref_cuda_time):>16.2f} | {torch_time / pref_cuda_time:>9.2f}x")
    print(f"{'PyTorch 标准 3x3 (ReLU+Res)':<30} | {conv3x3_time:>13.4f} ms | {get_gflops(conv3x3_flops, conv3x3_time):>16.2f} | {torch_time / conv3x3_time:>9.2f}x")
    print(f"{'PyTorch 标准 1x1 (ReLU+Res)':<30} | {conv1x1_time:>13.4f} ms | {get_gflops(conv1x1_flops, conv1x1_time):>16.2f} | {torch_time / conv1x1_time:>9.2f}x")
    print("-" * 90)
    print(f"PREF 内核相对于原内核的提升: {cuda_time / pref_cuda_time:.2f}x")
    print("-" * 70)

if __name__ == "__main__":
    benchmark_speed()
