#include <torch/extension.h>

// 声明在 evolution_kernel.cu 中定义的函数
at::Tensor evolution_cuda_forward(at::Tensor input, at::Tensor weights, at::Tensor biases);
std::vector<at::Tensor> evolution_cuda_backward(at::Tensor grad_y, at::Tensor x, at::Tensor weights);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &evolution_cuda_forward, "Evolution optimized forward (CUDA)");
    m.def("backward", &evolution_cuda_backward, "Evolution optimized backward (CUDA)");
}
