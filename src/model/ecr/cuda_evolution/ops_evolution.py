import os
import sys
import torch
from torch.utils.cpp_extension import load

# 获取当前脚本所在目录
CUDA_DIR = os.path.dirname(os.path.abspath(__file__))
build_path = os.path.join(CUDA_DIR, "build")

# 尝试优先加载预编译的 .pyd 文件
evolution_cuda = None
try:
    if build_path not in sys.path:
        sys.path.append(build_path)
    import evolution_cuda as evolution_cuda
except ImportError:
    # 如果预编译加载失败，则回退到 JIT 编译模式
    # 显式创建 build 目录，防止 torch.utils.file_baton 报错
    os.makedirs(build_path, exist_ok=True)
    evolution_cuda = load(
        name="evolution_cuda",
        sources=[
            os.path.join(CUDA_DIR, "evolution_bind.cpp"),
            os.path.join(CUDA_DIR, "evolution_kernel.cu")
        ],
        build_directory=build_path,
        verbose=False
    )

class EvolutionFunction(torch.autograd.Function):
    """
    极致优化的演化算子 (Fused ReLU + Depthwise Conv + Bias + Residual)
    支持自动求导
    """
    @staticmethod
    def forward(ctx, x, weights, biases):
        # 确保输入是连续的
        x = x.contiguous()
        weights = weights.contiguous()
        biases = biases.contiguous()
        
        # 调用 CUDA 前向内核
        output = evolution_cuda.forward(x, weights, biases)
        
        # 保存用于反向传播的张量
        ctx.save_for_backward(x, weights)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # 获取前向传播保存的张量
        x, weights = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        
        # 调用 CUDA 反向内核
        # 返回值: [grad_x, grad_weights, grad_biases]
        grads = evolution_cuda.backward(grad_output, x, weights)
        
        return grads[0], grads[1], grads[2]

def evolution_op(x, weights, biases):
    """
    演化算子的快捷调用函数
    """
    return EvolutionFunction.apply(x, weights, biases)

class EvolutionLayer(torch.nn.Module):
    """
    Evolution Optimized Module.
    封装了融合演化算子的权重和偏置。
    """
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        # weights: (channels, 3, 3) -> depthwise convolution weights
        self.weights = torch.nn.Parameter(torch.empty(channels, 3, 3))
        # biases: (channels)
        self.biases = torch.nn.Parameter(torch.empty(channels))
        self._reset_parameters()

    def _reset_parameters(self):
        # 使用 Kaiming 初始化
        torch.nn.init.kaiming_uniform_(self.weights, a=0.2)
        torch.nn.init.zeros_(self.biases)

    def forward(self, x):
        # 算子内部已包含残差连接 (x + conv(x))
        return evolution_op(x, self.weights, self.biases)
