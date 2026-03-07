import torch
import torch.nn.functional as F
import os
import time
from torch.utils.cpp_extension import load

# 获取当前脚本所在目录
CUDA_DIR = os.path.dirname(os.path.abspath(__file__))

class CudaOpTester:
    """
    通用的 CUDA 算子编译与测试框架
    站在架构师视角，将算子生命周期（编译、执行、验证）抽象化
    """
    
    def __init__(self, name, sources, verbose=True):
        self.name = name
        # 确保路径为绝对路径
        self.sources = [os.path.join(CUDA_DIR, s) for s in sources]
        self.verbose = verbose
        self.module = None

    def compile(self):
        """
        动态编译 CUDA 算子
        """
        print(f"\n>>> [Step 1] 正在编译算子: {self.name} ...")
        start_time = time.time()
        try:
            # 使用 TORCH_EXTENSION_NAME 机制进行即时编译 (JIT)
            self.module = load(
                name=f"{self.name}_lib",
                sources=self.sources,
                verbose=self.verbose
            )
            print(f"状态: [OK] 编译成功 (耗时 {time.time() - start_time:.2f}s)")
            return True
        except Exception as e:
            print(f"状态: [FAILED] 编译失败: {e}")
            return False

    def verify(self, input_args, ref_fn, cuda_fn_name="forward", tol=1e-4):
        """
        验证 CUDA 实现与 PyTorch 参考实现的一致性
        input_args: 传给算子的参数列表 (Tensor 等)
        ref_fn: PyTorch 参考实现函数
        cuda_fn_name: 编译后的模块中对应的函数名
        """
        if self.module is None:
            print("错误: 模块未编译，请先调用 compile()")
            return False

        print(f"\n>>> [Step 2] 正在验证算子逻辑: {self.name} ({cuda_fn_name}) ...")
        
        # 将输入移动到 GPU 准备测试
        cuda_args = [arg.cuda() if isinstance(arg, torch.Tensor) else arg for arg in input_args]
        
        with torch.no_grad():
            # 1. 执行 CUDA 版本
            cuda_fn = getattr(self.module, cuda_fn_name)
            output_cuda = cuda_fn(*cuda_args)
            
            # 2. 执行 PyTorch 版本 (参考实现)
            output_ref = ref_fn(*cuda_args)
            
            # 3. 计算误差 (Max Absolute Difference)
            diff = (output_cuda - output_ref).abs().max().item()
            print(f"最大绝对误差 (Max Abs Diff): {diff:.6e}")
            
            if diff < tol:
                print(f"状态: [PASS] 逻辑验证通过 (阈值={tol})")
                return True
            else:
                print(f"状态: [FAIL] 结果差异过大，请检查内核实现")
                return False

# =============================================================================
# 算子测试用例定义
# =============================================================================

# --- 1. Evolution8 (8层连续演化) ---

def pytorch_evolution8_forward(input_tensor, weights, biases):
    """8层连续演化的 PyTorch 参考实现"""
    x = input_tensor
    C = input_tensor.size(1)
    for i in range(8):
        # 演化逻辑：y = x + conv(relu(x))
        w = weights[:, i, :, :].unsqueeze(1) # [C, 1, 3, 3]
        b = biases[:, i]
        activated_x = F.relu(x)
        conv_res = F.conv2d(activated_x, w, bias=b, padding=1, groups=C)
        x = x + conv_res
    return x

def test_evolution8():
    tester = CudaOpTester(
        name="evolution8",
        sources=["evolution_v8_bind.cpp", "evolution_v8_kernel.cu"]
    )
    if not tester.compile(): return False
    
    # 构造输入 (B, C, H, W)
    B, C, H, W = 2, 16, 64, 64
    input_tensor = torch.randn(B, C, H, W)
    # weights: [C, 8, 3, 3], biases: [C, 8]
    weights = torch.randn(C, 8, 3, 3) * 0.02
    biases = torch.randn(C, 8) * 0.01
    
    return tester.verify([input_tensor, weights, biases], pytorch_evolution8_forward)

# --- 2. EvolutionPref (极致优化单层演化) ---

def pytorch_evolution_pref_forward(input_tensor, weights, biases):
    """极致优化版演化层的 PyTorch 参考实现"""
    C = input_tensor.size(1)
    # weights: [C, 3, 3] -> [C, 1, 3, 3] for depthwise conv
    w = weights.view(C, 1, 3, 3)
    activated_x = F.relu(input_tensor)
    conv_res = F.conv2d(activated_x, w, bias=biases, padding=1, groups=C)
    return input_tensor + conv_res

def test_evolution_pref():
    tester = CudaOpTester(
        name="evolution_pref",
        sources=["evolution_v1_bind.cpp", "evolution_v1_kernel.cu"]
    )
    if not tester.compile(): return False
    
    # 构造输入
    B, C, H, W = 2, 16, 64, 64
    input_tensor = torch.randn(B, C, H, W)
    # weights: [C, 3, 3], biases: [C]
    weights = torch.randn(C, 3, 3) * 0.02
    biases = torch.randn(C) * 0.01
    
    return tester.verify([input_tensor, weights, biases], pytorch_evolution_pref_forward)

# =============================================================================
# 主程序入口
# =============================================================================

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("\n[!] 错误: 未检测到可用 GPU，CUDA 算子测试需要 GPU 环境")
    else:
        print("\n" + "="*60)
        print("  CUDA 算子通用编译与验证工具 (JIT Mode)")
        print("="*60)
        
        # 记录各算子测试结果
        summary = []
        
        # 运行 Evolution8 测试
        summary.append(("Evolution8", test_evolution8()))
        
        # 运行 EvolutionPref 测试
        summary.append(("EvolutionPref", test_evolution_pref()))
        
        # 打印最终汇总报告
        print("\n" + "="*60)
        print(f"{'算子名称':<20} | {'测试结果':<10}")
        print("-" * 60)
        for name, success in summary:
            status = "PASS" if success else "FAILED"
            print(f"{name:<20} | {status:<10}")
        print("="*60 + "\n")
