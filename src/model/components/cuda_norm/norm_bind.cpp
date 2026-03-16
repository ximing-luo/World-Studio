#include <torch/extension.h>
#include <vector>

// Forward declarations
void rms_norm_2d_fwd_cuda(at::Tensor input, at::Tensor weight, at::Tensor output, at::Tensor inv_rms, float eps);
void rms_norm_2d_bwd_cuda(at::Tensor grad_output, at::Tensor input, at::Tensor weight, at::Tensor inv_rms, at::Tensor grad_input, at::Tensor grad_weight);
void layer_norm_2d_fwd_cuda(at::Tensor input, at::Tensor weight, at::Tensor bias, at::Tensor output, at::Tensor mean, at::Tensor inv_var, float eps);
void layer_norm_2d_bwd_cuda(at::Tensor grad_output, at::Tensor input, at::Tensor weight, at::Tensor mean, at::Tensor inv_var, at::Tensor grad_input, at::Tensor grad_weight, at::Tensor grad_bias);

// C++ wrappers
std::vector<at::Tensor> rms_norm_2d_fwd(at::Tensor input, at::Tensor weight, float eps) {
    auto output = at::empty_like(input);
    auto inv_rms = at::empty({input.size(0), 1, input.size(2), input.size(3)}, input.options().dtype(at::kFloat));
    rms_norm_2d_fwd_cuda(input, weight, output, inv_rms, eps);
    return {output, inv_rms};
}

std::vector<at::Tensor> rms_norm_2d_bwd(at::Tensor grad_output, at::Tensor input, at::Tensor weight, at::Tensor inv_rms) {
    auto grad_input = at::empty_like(input);
    auto grad_weight = at::zeros({weight.size(0)}, input.options().dtype(at::kFloat));
    rms_norm_2d_bwd_cuda(grad_output, input, weight, inv_rms, grad_input, grad_weight);
    return {grad_input, grad_weight};
}

std::vector<at::Tensor> layer_norm_2d_fwd(at::Tensor input, at::Tensor weight, at::Tensor bias, float eps) {
    auto output = at::empty_like(input);
    auto mean = at::empty({input.size(0), 1, input.size(2), input.size(3)}, input.options().dtype(at::kFloat));
    auto inv_var = at::empty({input.size(0), 1, input.size(2), input.size(3)}, input.options().dtype(at::kFloat));
    layer_norm_2d_fwd_cuda(input, weight, bias, output, mean, inv_var, eps);
    return {output, mean, inv_var};
}

std::vector<at::Tensor> layer_norm_2d_bwd(at::Tensor grad_output, at::Tensor input, at::Tensor weight, at::Tensor mean, at::Tensor inv_var) {
    auto grad_input = at::empty_like(input);
    auto grad_weight = at::zeros({weight.size(0)}, input.options().dtype(at::kFloat));
    auto grad_bias = at::zeros({weight.size(0)}, input.options().dtype(at::kFloat));
    layer_norm_2d_bwd_cuda(grad_output, input, weight, mean, inv_var, grad_input, grad_weight, grad_bias);
    return {grad_input, grad_weight, grad_bias};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rms_norm_2d_fwd", &rms_norm_2d_fwd, "RMSNorm2d forward (CUDA)");
    m.def("rms_norm_2d_bwd", &rms_norm_2d_bwd, "RMSNorm2d backward (CUDA)");
    m.def("layer_norm_2d_fwd", &layer_norm_2d_fwd, "LayerNorm2d forward (CUDA)");
    m.def("layer_norm_2d_bwd", &layer_norm_2d_bwd, "LayerNorm2d backward (CUDA)");
}
