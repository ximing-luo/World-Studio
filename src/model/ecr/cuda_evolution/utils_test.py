import torch
import torch.nn.functional as F
from ops_evolution import evolution_op
import time

def pytorch_evolution_ref(x, weights, biases):
    """
    PyTorch 参考实现: y = x + conv2d(relu(x), w) + b
    """
    C = x.size(1)
    w = weights.view(C, 1, 3, 3)
    activated_x = F.relu(x)
    conv_res = F.conv2d(activated_x, w, bias=biases, padding=1, groups=C)
    return x + conv_res

def benchmark_evolution():
    print(">>> 正在启动演化算子性能基准测试 (Evolution Op Benchmark) ...")
    
    # 设置测试参数 (使用典型的训练 Batch Size 和分辨率)
    B, C, H, W = 64, 128, 64, 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if device.type != "cuda":
        print("跳过测试: 未检测到 CUDA 设备")
        return

    # 初始化输入和参数
    x = torch.randn(B, C, H, W, device=device).requires_grad_(True)
    weights = (torch.randn(C, 3, 3, device=device) * 0.01).requires_grad_(True)
    biases = (torch.randn(C, device=device) * 0.01).requires_grad_(True)

    # 预热 (Warmup)
    print(f"正在预热 (Warmup)... [B={B}, C={C}, H={H}, W={W}]")
    for _ in range(20):
        y_ref = pytorch_evolution_ref(x, weights, biases)
        y_ref.sum().backward()
        y_cuda = evolution_op(x, weights, biases)
        y_cuda.sum().backward()
    torch.cuda.synchronize()

    iters = 100

    # 1. 测试 PyTorch 原生实现速度
    print(f"正在测试 PyTorch 原生实现 ({iters} 次迭代)...")
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(iters):
        # 清零梯度以模拟真实训练步骤 (可选，这里主要测算子时间)
        if x.grad is not None: x.grad.zero_()
        if weights.grad is not None: weights.grad.zero_()
        if biases.grad is not None: biases.grad.zero_()
        
        y = pytorch_evolution_ref(x, weights, biases)
        y.sum().backward()
    torch.cuda.synchronize()
    torch_total_time = (time.time() - start_time) * 1000 / iters

    # 2. 测试 CUDA 优化版本速度
    print(f"正在测试 CUDA 优化实现 ({iters} 次迭代)...")
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(iters):
        if x.grad is not None: x.grad.zero_()
        if weights.grad is not None: weights.grad.zero_()
        if biases.grad is not None: biases.grad.zero_()
        
        y = evolution_op(x, weights, biases)
        y.sum().backward()
    torch.cuda.synchronize()
    cuda_total_time = (time.time() - start_time) * 1000 / iters

    # 3. 输出结果
    print("\n" + "="*50)
    print(f"{'实现方式':<20} | {'平均耗时 (Forward+Backward)':<25}")
    print("-" * 50)
    print(f"{'PyTorch Native':<20} | {torch_total_time:>18.4f} ms")
    print(f"{'CUDA Optimized':<20} | {cuda_total_time:>18.4f} ms")
    print("-" * 50)
    speedup = torch_total_time / cuda_total_time
    print(f"加速比 (Speedup): {speedup:.2f}x")
    print("="*50 + "\n")

if __name__ == "__main__":
    benchmark_evolution()
