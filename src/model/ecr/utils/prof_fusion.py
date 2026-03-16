import torch
import torch.nn as nn
import torch.nn.functional as F
import time

class CrossScholarFusion(nn.Module):
    def __init__(self, in_channels, out_channels, latent_dim=None):
        super().__init__()
        if latent_dim is None:
            latent_dim = in_channels // 16
        if latent_dim < 8:
            latent_dim = 64
        self.latent_dim = latent_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        w_k = torch.randn(latent_dim, in_channels, 1, 1) * (latent_dim ** -0.5)
        self.W_K = nn.Parameter(w_k)
        
        w_q = torch.randn(out_channels, latent_dim, 1, 1) * (latent_dim ** -0.5)
        self.W_Q = nn.Parameter(w_q)

    def forward(self, x):
        # 移除 to(memory_format) 转换，因为它是一个同步操作，开销极大
        # 如果输入已经是 Channels Last，则直接运行
        x = F.conv2d(x, self.W_K)
        x = F.conv2d(x, self.W_Q)
        return x

def check_memory_format(x):
    if x.is_contiguous(memory_format=torch.channels_last):
        return "Channels Last (NHWC)"
    elif x.is_contiguous():
        return "Contiguous (NCHW)"
    else:
        return "Unknown/Other"

def benchmark_ecr_performance():
    if not torch.cuda.is_available():
        print("CUDA not available.")
        return

    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
    
    # 测试配置
    B, C, H, W = 64, 512, 32, 32
    L = 64 # Scholar Fusion 潜空间维度
    
    print("\n" + "="*80)
    print(f"FINAL PERFORMANCE COMPARISON | Input: ({B}, {C}, {H}, {W})")
    print("="*80)

    # 1. 准备模型
    model_dense = nn.Conv2d(C, C, 1, bias=False).to(device)
    model_scholar = CrossScholarFusion(C, C, latent_dim=L).to(device)
    
    # 2. 准备数据
    x_nchw = torch.randn(B, C, H, W, device=device)
    x_nhwc = x_nchw.to(memory_format=torch.channels_last)
    
    print(f"Testing with NCHW Input:")
    # 预热
    for _ in range(20):
        _ = model_dense(x_nchw)
        _ = model_scholar(x_nchw)
    torch.cuda.synchronize()

    def get_time(f, inp, iters=20):
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(iters):
            _ = f(inp)
        torch.cuda.synchronize()
        return (time.time() - start) / iters * 1000, None

    t_dense_nchw, _ = get_time(model_dense, x_nchw)
    out_dense_nchw = model_dense(x_nchw)
    t_scholar_nchw, _ = get_time(model_scholar, x_nchw)
    out_scholar_nchw = model_scholar(x_nchw)

    print(f"Standard 1x1: {t_dense_nchw:>12.4f} ms | {check_memory_format(out_dense_nchw)}")
    print(f"Scholar 1x1:  {t_scholar_nchw:>12.4f} ms | {check_memory_format(out_scholar_nchw)}")
    
    print(f"\nTesting with NHWC Input:")
    # # 预处理模型权重为 NHWC
    # model_dense_nhwc = nn.Conv2d(C, C, 1, bias=False).to(device).to(memory_format=torch.channels_last)
    # # 预热
    # for _ in range(20):
    #     _ = model_dense_nhwc(x_nhwc)
    #     _ = model_scholar(x_nhwc)
    # torch.cuda.synchronize()

    # t_dense_nhwc, _ = get_time(model_dense_nhwc, x_nhwc)
    # out_dense_nhwc = model_dense_nhwc(x_nhwc)
    # t_scholar_nhwc, _ = get_time(model_scholar, x_nhwc)
    # out_scholar_nhwc = model_scholar(x_nhwc)

    # print(f"Standard 1x1: {t_dense_nhwc:>12.4f} ms | {check_memory_format(out_dense_nhwc)}")
    # print(f"Scholar 1x1:  {t_scholar_nhwc:>12.4f} ms | {check_memory_format(out_scholar_nhwc)}")

    # print("-" * 80)
    # print(f"Best Speedup (NHWC): {t_dense_nhwc / t_scholar_nhwc:.2f}x")
    # print("="*80 + "\n")

if __name__ == "__main__":
    benchmark_ecr_performance()
