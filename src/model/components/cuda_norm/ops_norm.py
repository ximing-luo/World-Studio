import os
import torch
from torch.utils.cpp_extension import load

import sys
current_path = os.path.dirname(os.path.abspath(__file__))
build_path = os.path.join(current_path, "build")

# 尝试优先加载预编译的 .pyd 文件
norm_cuda = None
try:
    if build_path not in sys.path:
        sys.path.append(build_path)
    import norm_cuda_fast as norm_cuda
except ImportError:
    # 如果预编译加载失败，则回退到 JIT 编译模式
    # 显式创建 build 目录，因为 PyTorch 的 load 在创建 lock 文件时不会自动创建父目录
    os.makedirs(build_path, exist_ok=True)
    norm_cuda = load(
        name="norm_cuda_fast",
        sources=[
            os.path.join(current_path, "norm_bind.cpp"),
            os.path.join(current_path, "norm_kernel.cu"),
        ],
        build_directory=build_path,
        verbose=False
    )

class RMSNorm2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        w = weight.view(-1).contiguous()
        output, inv_rms = norm_cuda.rms_norm_2d_fwd(x, w, eps)
        ctx.save_for_backward(x, weight, inv_rms)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, inv_rms = ctx.saved_tensors
        w = weight.view(-1).contiguous()
        grad_input, grad_weight = norm_cuda.rms_norm_2d_bwd(
            grad_output.contiguous(), x, w, inv_rms
        )
        return grad_input, grad_weight.view_as(weight), None


class LayerNorm2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        w = weight.view(-1).contiguous()
        b = bias.view(-1).contiguous()
        output, mean, inv_var = norm_cuda.layer_norm_2d_fwd(x, w, b, eps)
        ctx.save_for_backward(x, weight, mean, inv_var)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, mean, inv_var = ctx.saved_tensors
        w = weight.view(-1).contiguous()
        grad_input, grad_weight, grad_bias = norm_cuda.layer_norm_2d_bwd(
            grad_output.contiguous(), x, w, mean, inv_var
        )
        return grad_input, grad_weight.view_as(weight), grad_bias.view_as(weight), None

def rms_norm_2d(x, weight, eps=1e-4):
    if not x.is_cuda:
        return x * torch.rsqrt(x.float().pow(2).mean(1, keepdim=True) + eps).type_as(x) * weight
    return RMSNorm2dFunction.apply(x, weight, eps)

def layer_norm_2d(x, weight, bias, eps=1e-6):
    if not x.is_cuda:
        mean = x.mean(1, keepdim=True)
        var = (x - mean).pow(2).mean(1, keepdim=True)
        x = (x - mean) * torch.rsqrt(var + eps)
        return x * weight + bias
    return LayerNorm2dFunction.apply(x, weight, bias, eps)
