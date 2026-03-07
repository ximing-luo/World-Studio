#include <torch/extension.h>

// 声明在 evolution_v1_kernel.cu 中定义的函数
// 使用 at::Tensor 并放在一行，减少 MSVC 对换行符和类型的潜在歧义
at::Tensor evolution_cuda_forward_pref(at::Tensor input, at::Tensor weights, at::Tensor biases);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &evolution_cuda_forward_pref, "Evolution optimized forward (CUDA)");
}
