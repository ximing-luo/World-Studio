import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def single_layer_evolution_kernel(
    x_ptr, y_ptr,
    w_ptr,  # 形状: [C, 3, 3] 展平
    b_ptr,  # 形状: [C] 展平
    B, C, H, W,
    stride_xb, stride_xc, stride_xh, stride_xw,
    stride_yb, stride_yc, stride_yh, stride_yw,
    BLOCK_SIZE: tl.constexpr,
):
    """
    单层演化内核 - 每个线程处理一个像素
    支持正确的残差连接和边界处理
    """
    # 程序ID
    pid = tl.program_id(0)

    # 计算总元素数
    total_elements = B * C * H * W

    # 创建偏移范围
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements

    # 解码4D索引
    idx = offsets
    w_idx = idx % W
    h_idx = (idx // W) % H
    c_idx = (idx // (H * W)) % C
    b_idx = idx // (C * H * W)

    # 计算指针
    x_ptrs = x_ptr + b_idx * stride_xb + c_idx * stride_xc + h_idx * stride_xh + w_idx * stride_xw
    y_ptrs = y_ptr + b_idx * stride_yb + c_idx * stride_yc + h_idx * stride_yh + w_idx * stride_yw

    # 加载输入
    x_val = tl.load(x_ptrs, mask=mask)
    identity = x_val  # 保存原始输入用于残差连接

    # 3x3卷积 - 对邻居应用ReLU
    conv_result = tl.zeros_like(x_val)

    # 加载当前通道的权重
    weight_base = w_ptr + c_idx * 9  # 第c通道

    # 遍历3x3邻域
    for dh in range(-1, 2):
        for dw in range(-1, 2):
            nh = h_idx + dh
            nw = w_idx + dw

            # 检查边界
            valid_h = (nh >= 0) & (nh < H)
            valid_w = (nw >= 0) & (nw < W)
            valid_mask = mask & valid_h & valid_w

            # 计算邻居指针
            neighbor_ptrs = x_ptr + b_idx * stride_xb + c_idx * stride_xc + nh * stride_xh + nw * stride_xw

            # 加载邻居像素并应用ReLU
            neighbor = tl.load(neighbor_ptrs, mask=valid_mask)
            neighbor = tl.where(neighbor > 0, neighbor, 0.0)

            # 加载权重
            weight_idx = (dh + 1) * 3 + (dw + 1)
            weight = tl.load(weight_base + weight_idx)
            conv_result = conv_result + neighbor * weight

    # 加偏置
    bias = tl.load(b_ptr + c_idx)  # 第c通道
    conv_result = conv_result + bias

    # 残差连接：原始输入 + 卷积结果
    y_val = identity + conv_result

    # 存储结果
    tl.store(y_ptrs, y_val, mask=mask)

class TritonECR(nn.Module):
    def __init__(self, channels, num_layers=8):
        super().__init__()
        self.channels = channels
        self.num_layers = num_layers
        self.weights = nn.Parameter(torch.randn(num_layers, channels, 3, 3) * 0.02)
        self.biases = nn.Parameter(torch.zeros(num_layers, channels))

    def forward(self, x):
        B, C, H, W = x.shape

        # 使用多次内核调用实现多层卷积
        current = x
        for layer in range(self.num_layers):
            # 为当前层分配输出
            y = torch.empty_like(current)

            # 获取当前层的权重和偏置
            layer_weights = self.weights[layer].view(-1)  # [C * 9]
            layer_biases = self.biases[layer].view(-1)    # [C]

            # 计算网格
            BLOCK_SIZE = 256
            total_elements = B * C * H * W
            grid = (triton.cdiv(total_elements, BLOCK_SIZE),)

            # 调用单层内核
            single_layer_evolution_kernel[grid](
                current, y, layer_weights, layer_biases,
                B, C, H, W,
                current.stride(0), current.stride(1), current.stride(2), current.stride(3),
                y.stride(0), y.stride(1), y.stride(2), y.stride(3),
                BLOCK_SIZE
            )

            # 当前输出作为下一层的输入
            current = y

        return current
