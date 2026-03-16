#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>

__device__ __forceinline__ float warp_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

__device__ __forceinline__ float block_sum(float v) {
    __shared__ float warp_buf[32];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    v = warp_sum(v);
    if (lane == 0) {
        warp_buf[warp] = v;
    }
    __syncthreads();
    if (warp == 0) {
        int warp_count = (blockDim.x + 31) >> 5;
        float out = (lane < warp_count) ? warp_buf[lane] : 0.0f;
        out = warp_sum(out);
        if (lane == 0) {
            warp_buf[0] = out;
        }
    }
    __syncthreads();
    return warp_buf[0];
}

template <typename T>
__global__ void rms_norm_fwd_scalar_kernel(
    const T* __restrict__ input,
    const T* __restrict__ weight,
    T* __restrict__ output,
    float* __restrict__ inv_rms,
    int B, int C, int H, int W, float eps) {
    int bhw = blockIdx.x;
    int HW = H * W;
    int total = B * HW;
    if (bhw >= total) return;
    int b = bhw / HW;
    int hw = bhw % HW;
    int base = b * C * HW + hw;
    float sum_sq = 0.0f;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float x = static_cast<float>(input[base + c * HW]);
        sum_sq += x * x;
    }
    float total_sq = block_sum(sum_sq);
    float r = rsqrtf(total_sq / C + eps);
    if (threadIdx.x == 0) {
        inv_rms[bhw] = r;
    }
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        int idx = base + c * HW;
        float x = static_cast<float>(input[idx]);
        float w = static_cast<float>(weight[c]);
        output[idx] = static_cast<T>(x * r * w);
    }
}

__global__ void rms_norm_fwd_vec4_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    float* __restrict__ inv_rms,
    int B, int C, int H, int W, float eps) {
    int HW = H * W;
    int total_bhw4 = (B * HW) >> 2;
    int bhw4 = blockIdx.x * blockDim.x + threadIdx.x;
    if (bhw4 >= total_bhw4) return;
    int bhw_base = bhw4 << 2;
    int b = bhw_base / HW;
    int hw = bhw_base % HW;
    int base = b * C * HW + hw;
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    for (int c = 0; c < C; ++c) {
        float4 x4 = reinterpret_cast<const float4*>(&input[base + c * HW])[0];
        s0 += x4.x * x4.x;
        s1 += x4.y * x4.y;
        s2 += x4.z * x4.z;
        s3 += x4.w * x4.w;
    }
    float r0 = rsqrtf(s0 / C + eps);
    float r1 = rsqrtf(s1 / C + eps);
    float r2 = rsqrtf(s2 / C + eps);
    float r3 = rsqrtf(s3 / C + eps);
    inv_rms[bhw_base + 0] = r0;
    inv_rms[bhw_base + 1] = r1;
    inv_rms[bhw_base + 2] = r2;
    inv_rms[bhw_base + 3] = r3;
    for (int c = 0; c < C; ++c) {
        float4 x4 = reinterpret_cast<const float4*>(&input[base + c * HW])[0];
        float w = weight[c];
        float4 o4;
        o4.x = x4.x * r0 * w;
        o4.y = x4.y * r1 * w;
        o4.z = x4.z * r2 * w;
        o4.w = x4.w * r3 * w;
        reinterpret_cast<float4*>(&output[base + c * HW])[0] = o4;
    }
}

template <typename T>
__global__ void rms_norm_bwd_input_scalar_kernel(
    const T* __restrict__ grad_output,
    const T* __restrict__ input,
    const T* __restrict__ weight,
    const float* __restrict__ inv_rms,
    T* __restrict__ grad_input,
    int B, int C, int H, int W) {
    int bhw = blockIdx.x;
    int HW = H * W;
    int total = B * HW;
    if (bhw >= total) return;
    int b = bhw / HW;
    int hw = bhw % HW;
    int base = b * C * HW + hw;
    float r = inv_rms[bhw];
    float dot_local = 0.0f;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        int idx = base + c * HW;
        float dy = static_cast<float>(grad_output[idx]);
        float x = static_cast<float>(input[idx]);
        float w = static_cast<float>(weight[c]);
        dot_local += dy * w * x;
    }
    float dot = block_sum(dot_local);
    float r3_over_c = (r * r * r) / static_cast<float>(C);
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        int idx = base + c * HW;
        float dy = static_cast<float>(grad_output[idx]);
        float x = static_cast<float>(input[idx]);
        float w = static_cast<float>(weight[c]);
        grad_input[idx] = static_cast<T>(dy * w * r - x * dot * r3_over_c);
    }
}

__global__ void rms_norm_bwd_input_vec4_kernel(
    const float* __restrict__ grad_output,
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ inv_rms,
    float* __restrict__ grad_input,
    int B, int C, int H, int W) {
    int HW = H * W;
    int total_bhw4 = (B * HW) >> 2;
    int bhw4 = blockIdx.x * blockDim.x + threadIdx.x;
    if (bhw4 >= total_bhw4) return;
    int bhw_base = bhw4 << 2;
    int b = bhw_base / HW;
    int hw = bhw_base % HW;
    int base = b * C * HW + hw;
    float r0 = inv_rms[bhw_base + 0];
    float r1 = inv_rms[bhw_base + 1];
    float r2 = inv_rms[bhw_base + 2];
    float r3 = inv_rms[bhw_base + 3];
    float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
    for (int c = 0; c < C; ++c) {
        float4 dy4 = reinterpret_cast<const float4*>(&grad_output[base + c * HW])[0];
        float4 x4 = reinterpret_cast<const float4*>(&input[base + c * HW])[0];
        float w = weight[c];
        d0 += dy4.x * w * x4.x;
        d1 += dy4.y * w * x4.y;
        d2 += dy4.z * w * x4.z;
        d3 += dy4.w * w * x4.w;
    }
    float k0 = (r0 * r0 * r0) / C;
    float k1 = (r1 * r1 * r1) / C;
    float k2 = (r2 * r2 * r2) / C;
    float k3 = (r3 * r3 * r3) / C;
    for (int c = 0; c < C; ++c) {
        float4 dy4 = reinterpret_cast<const float4*>(&grad_output[base + c * HW])[0];
        float4 x4 = reinterpret_cast<const float4*>(&input[base + c * HW])[0];
        float w = weight[c];
        float4 gx4;
        gx4.x = dy4.x * w * r0 - x4.x * d0 * k0;
        gx4.y = dy4.y * w * r1 - x4.y * d1 * k1;
        gx4.z = dy4.z * w * r2 - x4.z * d2 * k2;
        gx4.w = dy4.w * w * r3 - x4.w * d3 * k3;
        reinterpret_cast<float4*>(&grad_input[base + c * HW])[0] = gx4;
    }
}

template <typename T>
__global__ void rms_norm_grad_weight_partial_kernel(
    const T* __restrict__ grad_output,
    const T* __restrict__ input,
    const float* __restrict__ inv_rms,
    float* __restrict__ partial,
    int B, int C, int H, int W, int tiles) {
    int c = blockIdx.x;
    int tile = blockIdx.y;
    if (c >= C || tile >= tiles) return;
    int HW = H * W;
    int total = B * HW;
    int start = tile * blockDim.x;
    float local = 0.0f;
    int i = start + threadIdx.x;
    if (i < total) {
        int b = i / HW;
        int hw = i % HW;
        int idx = b * C * HW + c * HW + hw;
        local = static_cast<float>(grad_output[idx]) * static_cast<float>(input[idx]) * inv_rms[i];
    }
    float sum = block_sum(local);
    if (threadIdx.x == 0) {
        partial[tile * C + c] = sum;
    }
}

__global__ void reduce_partial_sum_kernel(
    const float* __restrict__ partial,
    float* __restrict__ out,
    int C, int tiles) {
    int c = blockIdx.x;
    if (c >= C) return;
    float local = 0.0f;
    for (int t = threadIdx.x; t < tiles; t += blockDim.x) {
        local += partial[t * C + c];
    }
    float sum = block_sum(local);
    if (threadIdx.x == 0) {
        out[c] = sum;
    }
}

template <typename T>
__global__ void layer_norm_fwd_scalar_kernel(
    const T* __restrict__ input,
    const T* __restrict__ weight,
    const T* __restrict__ bias,
    T* __restrict__ output,
    float* __restrict__ mean,
    float* __restrict__ inv_var,
    int B, int C, int H, int W, float eps) {
    int bhw = blockIdx.x;
    int HW = H * W;
    int total = B * HW;
    if (bhw >= total) return;
    int b = bhw / HW;
    int hw = bhw % HW;
    int base = b * C * HW + hw;
    float s = 0.0f;
    float s2 = 0.0f;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float x = static_cast<float>(input[base + c * HW]);
        s += x;
        s2 += x * x;
    }
    float sum = block_sum(s);
    float sum_sq = block_sum(s2);
    float m = sum / C;
    float var = fmaxf(sum_sq / C - m * m, 0.0f);
    float r = rsqrtf(var + eps);
    if (threadIdx.x == 0) {
        mean[bhw] = m;
        inv_var[bhw] = r;
    }
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        int idx = base + c * HW;
        float x = static_cast<float>(input[idx]);
        float w = static_cast<float>(weight[c]);
        float beta = static_cast<float>(bias[c]);
        output[idx] = static_cast<T>((x - m) * r * w + beta);
    }
}

__global__ void layer_norm_fwd_vec4_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    float* __restrict__ mean,
    float* __restrict__ inv_var,
    int B, int C, int H, int W, float eps) {
    int HW = H * W;
    int total_bhw4 = (B * HW) >> 2;
    int bhw4 = blockIdx.x * blockDim.x + threadIdx.x;
    if (bhw4 >= total_bhw4) return;
    int bhw_base = bhw4 << 2;
    int b = bhw_base / HW;
    int hw = bhw_base % HW;
    int base = b * C * HW + hw;
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    float q0 = 0.0f, q1 = 0.0f, q2 = 0.0f, q3 = 0.0f;
    for (int c = 0; c < C; ++c) {
        float4 x4 = reinterpret_cast<const float4*>(&input[base + c * HW])[0];
        s0 += x4.x; s1 += x4.y; s2 += x4.z; s3 += x4.w;
        q0 += x4.x * x4.x; q1 += x4.y * x4.y; q2 += x4.z * x4.z; q3 += x4.w * x4.w;
    }
    float m0 = s0 / C, m1 = s1 / C, m2 = s2 / C, m3 = s3 / C;
    float r0 = rsqrtf(fmaxf(q0 / C - m0 * m0, 0.0f) + eps);
    float r1 = rsqrtf(fmaxf(q1 / C - m1 * m1, 0.0f) + eps);
    float r2 = rsqrtf(fmaxf(q2 / C - m2 * m2, 0.0f) + eps);
    float r3 = rsqrtf(fmaxf(q3 / C - m3 * m3, 0.0f) + eps);
    mean[bhw_base + 0] = m0; mean[bhw_base + 1] = m1; mean[bhw_base + 2] = m2; mean[bhw_base + 3] = m3;
    inv_var[bhw_base + 0] = r0; inv_var[bhw_base + 1] = r1; inv_var[bhw_base + 2] = r2; inv_var[bhw_base + 3] = r3;
    for (int c = 0; c < C; ++c) {
        float4 x4 = reinterpret_cast<const float4*>(&input[base + c * HW])[0];
        float w = weight[c];
        float b0 = bias[c];
        float4 o4;
        o4.x = (x4.x - m0) * r0 * w + b0;
        o4.y = (x4.y - m1) * r1 * w + b0;
        o4.z = (x4.z - m2) * r2 * w + b0;
        o4.w = (x4.w - m3) * r3 * w + b0;
        reinterpret_cast<float4*>(&output[base + c * HW])[0] = o4;
    }
}

template <typename T>
__global__ void layer_norm_bwd_input_scalar_kernel(
    const T* __restrict__ grad_output,
    const T* __restrict__ input,
    const T* __restrict__ weight,
    const float* __restrict__ mean,
    const float* __restrict__ inv_var,
    T* __restrict__ grad_input,
    int B, int C, int H, int W) {
    int bhw = blockIdx.x;
    int HW = H * W;
    int total = B * HW;
    if (bhw >= total) return;
    int b = bhw / HW;
    int hw = bhw % HW;
    int base = b * C * HW + hw;
    float m = mean[bhw];
    float r = inv_var[bhw];
    float s1_local = 0.0f;
    float s2_local = 0.0f;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        int idx = base + c * HW;
        float x = static_cast<float>(input[idx]);
        float dy = static_cast<float>(grad_output[idx]);
        float w = static_cast<float>(weight[c]);
        float g = dy * w;
        float xh = (x - m) * r;
        s1_local += g;
        s2_local += g * xh;
    }
    float s1 = block_sum(s1_local);
    float s2 = block_sum(s2_local);
    float inv_c = 1.0f / C;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        int idx = base + c * HW;
        float x = static_cast<float>(input[idx]);
        float dy = static_cast<float>(grad_output[idx]);
        float w = static_cast<float>(weight[c]);
        float g = dy * w;
        float xh = (x - m) * r;
        float dx = (g - s1 * inv_c - xh * s2 * inv_c) * r;
        grad_input[idx] = static_cast<T>(dx);
    }
}

__global__ void layer_norm_bwd_input_vec4_kernel(
    const float* __restrict__ grad_output,
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ mean,
    const float* __restrict__ inv_var,
    float* __restrict__ grad_input,
    int B, int C, int H, int W) {
    int HW = H * W;
    int total_bhw4 = (B * HW) >> 2;
    int bhw4 = blockIdx.x * blockDim.x + threadIdx.x;
    if (bhw4 >= total_bhw4) return;
    int bhw_base = bhw4 << 2;
    int b = bhw_base / HW;
    int hw = bhw_base % HW;
    int base = b * C * HW + hw;
    float m0 = mean[bhw_base + 0], m1 = mean[bhw_base + 1], m2 = mean[bhw_base + 2], m3 = mean[bhw_base + 3];
    float r0 = inv_var[bhw_base + 0], r1 = inv_var[bhw_base + 1], r2 = inv_var[bhw_base + 2], r3 = inv_var[bhw_base + 3];
    float s10 = 0.0f, s11 = 0.0f, s12 = 0.0f, s13 = 0.0f;
    float s20 = 0.0f, s21 = 0.0f, s22 = 0.0f, s23 = 0.0f;
    for (int c = 0; c < C; ++c) {
        float4 dy4 = reinterpret_cast<const float4*>(&grad_output[base + c * HW])[0];
        float4 x4 = reinterpret_cast<const float4*>(&input[base + c * HW])[0];
        float w = weight[c];
        float g0 = dy4.x * w, g1 = dy4.y * w, g2 = dy4.z * w, g3 = dy4.w * w;
        float xh0 = (x4.x - m0) * r0, xh1 = (x4.y - m1) * r1, xh2 = (x4.z - m2) * r2, xh3 = (x4.w - m3) * r3;
        s10 += g0; s11 += g1; s12 += g2; s13 += g3;
        s20 += g0 * xh0; s21 += g1 * xh1; s22 += g2 * xh2; s23 += g3 * xh3;
    }
    float inv_c = 1.0f / C;
    for (int c = 0; c < C; ++c) {
        float4 dy4 = reinterpret_cast<const float4*>(&grad_output[base + c * HW])[0];
        float4 x4 = reinterpret_cast<const float4*>(&input[base + c * HW])[0];
        float w = weight[c];
        float g0 = dy4.x * w, g1 = dy4.y * w, g2 = dy4.z * w, g3 = dy4.w * w;
        float xh0 = (x4.x - m0) * r0, xh1 = (x4.y - m1) * r1, xh2 = (x4.z - m2) * r2, xh3 = (x4.w - m3) * r3;
        float4 dx4;
        dx4.x = (g0 - s10 * inv_c - xh0 * s20 * inv_c) * r0;
        dx4.y = (g1 - s11 * inv_c - xh1 * s21 * inv_c) * r1;
        dx4.z = (g2 - s12 * inv_c - xh2 * s22 * inv_c) * r2;
        dx4.w = (g3 - s13 * inv_c - xh3 * s23 * inv_c) * r3;
        reinterpret_cast<float4*>(&grad_input[base + c * HW])[0] = dx4;
    }
}

template <typename T>
__global__ void layer_norm_grad_param_partial_kernel(
    const T* __restrict__ grad_output,
    const T* __restrict__ input,
    const float* __restrict__ mean,
    const float* __restrict__ inv_var,
    float* __restrict__ partial_w,
    float* __restrict__ partial_b,
    int B, int C, int H, int W, int tiles) {
    int c = blockIdx.x;
    int tile = blockIdx.y;
    if (c >= C || tile >= tiles) return;
    int HW = H * W;
    int total = B * HW;
    int start = tile * blockDim.x;
    float lw = 0.0f;
    float lb = 0.0f;
    int i = start + threadIdx.x;
    if (i < total) {
        int b = i / HW;
        int hw = i % HW;
        int idx = b * C * HW + c * HW + hw;
        float dy = static_cast<float>(grad_output[idx]);
        float x = static_cast<float>(input[idx]);
        float xh = (x - mean[i]) * inv_var[i];
        lw = dy * xh;
        lb = dy;
    }
    float sw = block_sum(lw);
    float sb = block_sum(lb);
    if (threadIdx.x == 0) {
        partial_w[tile * C + c] = sw;
        partial_b[tile * C + c] = sb;
    }
}

void rms_norm_2d_fwd_cuda(at::Tensor input, at::Tensor weight, at::Tensor output, at::Tensor inv_rms, float eps) {
    int B = input.size(0), C = input.size(1), H = input.size(2), W = input.size(3);
    int total = B * H * W;
    bool can_vec4 = input.scalar_type() == at::kFloat && (W % 4 == 0) &&
        ((reinterpret_cast<uintptr_t>(input.data_ptr()) & 15) == 0) &&
        ((reinterpret_cast<uintptr_t>(output.data_ptr()) & 15) == 0);
    if (can_vec4) {
        int threads = 256;
        int blocks = ((total >> 2) + threads - 1) / threads;
        rms_norm_fwd_vec4_kernel<<<blocks, threads>>>(
            input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(),
            inv_rms.data_ptr<float>(), B, C, H, W, eps);
        return;
    }
    int threads = 256;
    int blocks = total;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "rms_norm_fwd_scalar_kernel", ([&] {
        rms_norm_fwd_scalar_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(), weight.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
            inv_rms.data_ptr<float>(), B, C, H, W, eps);
    }));
}

void rms_norm_2d_bwd_cuda(
    at::Tensor grad_output, at::Tensor input, at::Tensor weight, at::Tensor inv_rms,
    at::Tensor grad_input, at::Tensor grad_weight) {
    int B = input.size(0), C = input.size(1), H = input.size(2), W = input.size(3);
    int total = B * H * W;
    bool can_vec4 = input.scalar_type() == at::kFloat && (W % 4 == 0) &&
        ((reinterpret_cast<uintptr_t>(input.data_ptr()) & 15) == 0) &&
        ((reinterpret_cast<uintptr_t>(grad_output.data_ptr()) & 15) == 0) &&
        ((reinterpret_cast<uintptr_t>(grad_input.data_ptr()) & 15) == 0);
    if (can_vec4) {
        int threads = 256;
        int blocks = ((total >> 2) + threads - 1) / threads;
        rms_norm_bwd_input_vec4_kernel<<<blocks, threads>>>(
            grad_output.data_ptr<float>(), input.data_ptr<float>(), weight.data_ptr<float>(),
            inv_rms.data_ptr<float>(), grad_input.data_ptr<float>(), B, C, H, W);
    } else {
        int threads = 256;
        int blocks = total;
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "rms_norm_bwd_input_scalar_kernel", ([&] {
            rms_norm_bwd_input_scalar_kernel<scalar_t><<<blocks, threads>>>(
                grad_output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(), weight.data_ptr<scalar_t>(),
                inv_rms.data_ptr<float>(), grad_input.data_ptr<scalar_t>(), B, C, H, W);
        }));
    }
    int reduce_threads = 256;
    int tiles = (total + reduce_threads - 1) / reduce_threads;
    auto partial = torch::zeros({tiles, C}, grad_weight.options().dtype(at::kFloat));
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "rms_norm_grad_weight_partial_kernel", ([&] {
        rms_norm_grad_weight_partial_kernel<scalar_t><<<dim3(C, tiles), reduce_threads>>>(
            grad_output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(), inv_rms.data_ptr<float>(),
            partial.data_ptr<float>(), B, C, H, W, tiles);
    }));
    reduce_partial_sum_kernel<<<C, 256>>>(partial.data_ptr<float>(), grad_weight.data_ptr<float>(), C, tiles);
}

void layer_norm_2d_fwd_cuda(
    at::Tensor input, at::Tensor weight, at::Tensor bias, at::Tensor output,
    at::Tensor mean, at::Tensor inv_var, float eps) {
    int B = input.size(0), C = input.size(1), H = input.size(2), W = input.size(3);
    int total = B * H * W;
    bool can_vec4 = input.scalar_type() == at::kFloat && (W % 4 == 0) &&
        ((reinterpret_cast<uintptr_t>(input.data_ptr()) & 15) == 0) &&
        ((reinterpret_cast<uintptr_t>(output.data_ptr()) & 15) == 0);
    if (can_vec4) {
        int threads = 256;
        int blocks = ((total >> 2) + threads - 1) / threads;
        layer_norm_fwd_vec4_kernel<<<blocks, threads>>>(
            input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), output.data_ptr<float>(),
            mean.data_ptr<float>(), inv_var.data_ptr<float>(), B, C, H, W, eps);
        return;
    }
    int threads = 256;
    int blocks = total;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "layer_norm_fwd_scalar_kernel", ([&] {
        layer_norm_fwd_scalar_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(), weight.data_ptr<scalar_t>(), bias.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
            mean.data_ptr<float>(), inv_var.data_ptr<float>(), B, C, H, W, eps);
    }));
}

void layer_norm_2d_bwd_cuda(
    at::Tensor grad_output, at::Tensor input, at::Tensor weight, at::Tensor mean, at::Tensor inv_var,
    at::Tensor grad_input, at::Tensor grad_weight, at::Tensor grad_bias) {
    int B = input.size(0), C = input.size(1), H = input.size(2), W = input.size(3);
    int total = B * H * W;
    bool can_vec4 = input.scalar_type() == at::kFloat && (W % 4 == 0) &&
        ((reinterpret_cast<uintptr_t>(input.data_ptr()) & 15) == 0) &&
        ((reinterpret_cast<uintptr_t>(grad_output.data_ptr()) & 15) == 0) &&
        ((reinterpret_cast<uintptr_t>(grad_input.data_ptr()) & 15) == 0);
    if (can_vec4) {
        int threads = 256;
        int blocks = ((total >> 2) + threads - 1) / threads;
        layer_norm_bwd_input_vec4_kernel<<<blocks, threads>>>(
            grad_output.data_ptr<float>(), input.data_ptr<float>(), weight.data_ptr<float>(),
            mean.data_ptr<float>(), inv_var.data_ptr<float>(), grad_input.data_ptr<float>(), B, C, H, W);
    } else {
        int threads = 256;
        int blocks = total;
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "layer_norm_bwd_input_scalar_kernel", ([&] {
            layer_norm_bwd_input_scalar_kernel<scalar_t><<<blocks, threads>>>(
                grad_output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(), weight.data_ptr<scalar_t>(),
                mean.data_ptr<float>(), inv_var.data_ptr<float>(), grad_input.data_ptr<scalar_t>(), B, C, H, W);
        }));
    }
    int reduce_threads = 256;
    int tiles = (total + reduce_threads - 1) / reduce_threads;
    auto partial_w = torch::zeros({tiles, C}, grad_weight.options().dtype(at::kFloat));
    auto partial_b = torch::zeros({tiles, C}, grad_bias.options().dtype(at::kFloat));
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "layer_norm_grad_param_partial_kernel", ([&] {
        layer_norm_grad_param_partial_kernel<scalar_t><<<dim3(C, tiles), reduce_threads>>>(
            grad_output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(), mean.data_ptr<float>(), inv_var.data_ptr<float>(),
            partial_w.data_ptr<float>(), partial_b.data_ptr<float>(), B, C, H, W, tiles);
    }));
    reduce_partial_sum_kernel<<<C, 256>>>(partial_w.data_ptr<float>(), grad_weight.data_ptr<float>(), C, tiles);
    reduce_partial_sum_kernel<<<C, 256>>>(partial_b.data_ptr<float>(), grad_bias.data_ptr<float>(), C, tiles);
}
