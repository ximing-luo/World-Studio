#include <torch/extension.h>

// 声明在 evolution_v8_kernel.cu 中定义的函数
at::Tensor evolution8_cuda_forward(at::Tensor input, at::Tensor weights, at::Tensor biases);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &evolution8_cuda_forward, "Evolution 8-layer optimized forward (CUDA)");
}
