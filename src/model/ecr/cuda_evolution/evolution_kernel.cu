#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// -----------------------------------------------------------------------------
// Evolution Kernel (Extreme Optimized Version with float4 Vectorization)
// Strategies:
// 1. Memory Throughput (float4): Uses float4 for vectorized memory access (128-bit loads/stores).
// 2. Register Tiling: Uses 3x6 registers to cache local input data, avoiding shared memory sync overhead.
// 3. Weights Caching: Weights are loaded once into registers per thread.
// 4. Operator Fusion: ReLU + Depthwise Conv + Bias + Residual in a single kernel pass.
// -----------------------------------------------------------------------------

__global__ void evolution_kernel(
    const float* __restrict__ x, 
    const float* __restrict__ weights, 
    const float* __restrict__ biases, 
    float* __restrict__ y,
    int B, int C, int H, int W) {

    // Index mapping: one thread handles 4 horizontal pixels along the W dimension.
    const int b_c = blockIdx.x; // (Batch * Channel) index mapped to x to avoid z-limit
    const int c = b_c % C;      // Channel index for weight selection

    const int ty = blockIdx.y * blockDim.y + threadIdx.y;
    const int tx4 = (blockIdx.z * blockDim.x + threadIdx.x) * 4; // W mapped to z

    // Boundary check for H and W
    if (ty >= H || tx4 >= W || b_c >= B * C) {
        return;
    }

    // Load weights into registers (3x3 kernel = 9 elements)
    float w[9];
    const float* w_ptr = weights + (c * 9);
    #pragma unroll
    for (int i = 0; i < 9; ++i) {
        w[i] = w_ptr[i];
    }

    const float bias = biases[c];

    // Each thread processes 4 output pixels.
    // For a 3x3 kernel, it needs a 3x(4+2)=3x6 input window.
    float row_data[3][6]; 

    // Load the 3x6 window from global memory to registers
    #pragma unroll
    for (int i = 0; i < 3; ++i) {
        const int cur_h = ty + i - 1;
        if (cur_h < 0 || cur_h >= H) {
            #pragma unroll
            for (int j = 0; j < 6; ++j) {
                row_data[i][j] = 0.0f;
            }
        } else {
            const int row_base = (b_c * H + cur_h) * W;
            
            // Optimization: Vectorized Load for middle 4 pixels (if aligned and within bounds)
            if (tx4 + 3 < W) {
                // Load tx4 to tx4+3 using float4
                float4 val4 = reinterpret_cast<const float4*>(&x[row_base + tx4])[0];
                
                // Handle left boundary (tx4-1)
                float left_val = (tx4 > 0) ? x[row_base + tx4 - 1] : 0.0f;
                // Handle right boundary (tx4+4)
                float right_val = (tx4 + 4 < W) ? x[row_base + tx4 + 4] : 0.0f;

                // Store in register tile and fuse ReLU
                row_data[i][0] = (left_val > 0.0f) ? left_val : 0.0f;
                row_data[i][1] = (val4.x > 0.0f) ? val4.x : 0.0f;
                row_data[i][2] = (val4.y > 0.0f) ? val4.y : 0.0f;
                row_data[i][3] = (val4.z > 0.0f) ? val4.z : 0.0f;
                row_data[i][4] = (val4.w > 0.0f) ? val4.w : 0.0f;
                row_data[i][5] = (right_val > 0.0f) ? right_val : 0.0f;
            } else {
                // Fallback for non-multiple of 4 at the very edge of W
                #pragma unroll
                for (int j = 0; j < 6; ++j) {
                    const int cur_w = tx4 + j - 1;
                    if (cur_w >= 0 && cur_w < W) {
                        float val = x[row_base + cur_w];
                        row_data[i][j] = (val > 0.0f) ? val : 0.0f;
                    } else {
                        row_data[i][j] = 0.0f;
                    }
                }
            }
        }
    }

    // Output results buffer
    float res[4];
    #pragma unroll
    for (int k = 0; k < 4; ++k) res[k] = 0.0f;

    // Compute convolution for up to 4 horizontal output points
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        if (tx4 + k < W) {
            #pragma unroll
            for (int i = 0; i < 3; ++i) {
                #pragma unroll
                for (int j = 0; j < 3; ++j) {
                    res[k] += row_data[i][k + j] * w[i * 3 + j];
                }
            }
        }
    }

    // Output and Residual connection
    const int out_row_base = (b_c * H + ty) * W;
    if (tx4 + 3 < W) {
        // Vectorized load for Residual (x)
        float4 residual4 = reinterpret_cast<const float4*>(&x[out_row_base + tx4])[0];
        
        // Final fused results
        float4 out4;
        out4.x = res[0] + bias + residual4.x;
        out4.y = res[1] + bias + residual4.y;
        out4.z = res[2] + bias + residual4.z;
        out4.w = res[3] + bias + residual4.w;

        // Vectorized store for output (y)
        reinterpret_cast<float4*>(&y[out_row_base + tx4])[0] = out4;
    } else {
        // Scalar fallback for boundary pixels
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int out_w = tx4 + k;
            if (out_w < W) {
                const int idx = out_row_base + out_w;
                y[idx] = res[k] + bias + x[idx];
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Evolution Backward Kernels
// -----------------------------------------------------------------------------

__global__ void evolution_backward_input_kernel(
    const float* __restrict__ grad_y,
    const float* __restrict__ x,
    const float* __restrict__ weights,
    float* __restrict__ grad_x,
    int B, int C, int H, int W) {

    const int b_c = blockIdx.x;
    const int c = b_c % C;
    const int ty = blockIdx.y * blockDim.y + threadIdx.y;
    const int tx4 = (blockIdx.z * blockDim.x + threadIdx.x) * 4;

    if (ty >= H || tx4 >= W || b_c >= B * C) return;

    // Load flipped weights for transposed convolution (depthwise)
    float w_flipped[9];
    const float* w_ptr = weights + (c * 9);
    #pragma unroll
    for (int i = 0; i < 3; ++i) {
        #pragma unroll
        for (int j = 0; j < 3; ++j) {
            w_flipped[i * 3 + j] = w_ptr[(2 - i) * 3 + (2 - j)];
        }
    }

    // Load grad_y window (3x6) for vectorized processing
    float gy_data[3][6];
    #pragma unroll
    for (int i = 0; i < 3; ++i) {
        const int cur_h = ty + i - 1;
        if (cur_h < 0 || cur_h >= H) {
            #pragma unroll
            for (int j = 0; j < 6; ++j) gy_data[i][j] = 0.0f;
        } else {
            const int row_base = (b_c * H + cur_h) * W;
            if (tx4 + 3 < W) {
                float4 gy4 = reinterpret_cast<const float4*>(&grad_y[row_base + tx4])[0];
                float left_gy = (tx4 > 0) ? grad_y[row_base + tx4 - 1] : 0.0f;
                float right_gy = (tx4 + 4 < W) ? grad_y[row_base + tx4 + 4] : 0.0f;
                gy_data[i][0] = left_gy;
                gy_data[i][1] = gy4.x; gy_data[i][2] = gy4.y;
                gy_data[i][3] = gy4.z; gy_data[i][4] = gy4.w;
                gy_data[i][5] = right_gy;
            } else {
                #pragma unroll
                for (int j = 0; j < 6; ++j) {
                    const int cur_w = tx4 + j - 1;
                    gy_data[i][j] = (cur_w >= 0 && cur_w < W) ? grad_y[row_base + cur_w] : 0.0f;
                }
            }
        }
    }

    // Compute grad_relu_x via convolution
    float g_relu_x[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        if (tx4 + k < W) {
            #pragma unroll
            for (int i = 0; i < 3; ++i) {
                #pragma unroll
                for (int j = 0; j < 3; ++j) {
                    g_relu_x[k] += gy_data[i][k + j] * w_flipped[i * 3 + j];
                }
            }
        }
    }

    // Final grad_x = grad_y + g_relu_x * (x > 0)
    const int out_row_base = (b_c * H + ty) * W;
    if (tx4 + 3 < W) {
        // Optimization: Reuse grad_y from registers (gy_data[1][1..4]) instead of re-loading
        float4 gy4;
        gy4.x = gy_data[1][1];
        gy4.y = gy_data[1][2];
        gy4.z = gy_data[1][3];
        gy4.w = gy_data[1][4];

        float4 x4 = reinterpret_cast<const float4*>(&x[out_row_base + tx4])[0];
        float4 gx4;
        gx4.x = gy4.x + g_relu_x[0] * (x4.x > 0.0f ? 1.0f : 0.0f);
        gx4.y = gy4.y + g_relu_x[1] * (x4.y > 0.0f ? 1.0f : 0.0f);
        gx4.z = gy4.z + g_relu_x[2] * (x4.z > 0.0f ? 1.0f : 0.0f);
        gx4.w = gy4.w + g_relu_x[3] * (x4.w > 0.0f ? 1.0f : 0.0f);
        reinterpret_cast<float4*>(&grad_x[out_row_base + tx4])[0] = gx4;
    } else {
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int out_w = tx4 + k;
            if (out_w < W) {
                const int idx = out_row_base + out_w;
                grad_x[idx] = grad_y[idx] + g_relu_x[k] * (x[idx] > 0.0f ? 1.0f : 0.0f);
            }
        }
    }
}

__global__ void evolution_backward_param_kernel(
    const float* __restrict__ grad_y,
    const float* __restrict__ x,
    float* __restrict__ grad_weights,
    float* __restrict__ grad_biases,
    int B, int C, int H, int W) {

    const int c = blockIdx.x; 
    const int ty = blockIdx.y * blockDim.y + threadIdx.y;
    const int tx4 = (blockIdx.z * blockDim.x + threadIdx.x) * 4;

    if (ty >= H || tx4 >= W || c >= C) return;

    float local_gb = 0.0f;
    float local_gw[9] = {0.0f};

    // Optimization: Process 4 pixels per thread using vectorized loads
    for (int b = 0; b < B; ++b) {
        const int row_base = (b * C + c) * H * W;
        
        // 1. Load 4 grad_y values
        float gy[4] = {0.0f};
        if (tx4 + 3 < W) {
            float4 gy4 = reinterpret_cast<const float4*>(&grad_y[row_base + ty * W + tx4])[0];
            gy[0] = gy4.x; gy[1] = gy4.y; gy[2] = gy4.z; gy[3] = gy4.w;
        } else {
            for (int k = 0; k < 4; ++k) {
                if (tx4 + k < W) gy[k] = grad_y[row_base + ty * W + tx4 + k];
            }
        }

        // 2. Load x window (3x6) and fuse ReLU (same as forward kernel magic)
        float x_window[3][6];
        #pragma unroll
        for (int i = 0; i < 3; ++i) {
            const int cur_h = ty + i - 1;
            if (cur_h < 0 || cur_h >= H) {
                #pragma unroll
                for (int j = 0; j < 6; ++j) x_window[i][j] = 0.0f;
            } else {
                const int x_row_base = row_base + cur_h * W;
                if (tx4 + 3 < W) {
                    float4 x4 = reinterpret_cast<const float4*>(&x[x_row_base + tx4])[0];
                    float left_x = (tx4 > 0) ? x[x_row_base + tx4 - 1] : 0.0f;
                    float right_x = (tx4 + 4 < W) ? x[x_row_base + tx4 + 4] : 0.0f;
                    // Fused ReLU: param gradient is w.r.t. activated input
                    x_window[i][0] = (left_x > 0.0f) ? left_x : 0.0f;
                    x_window[i][1] = (x4.x > 0.0f) ? x4.x : 0.0f;
                    x_window[i][2] = (x4.y > 0.0f) ? x4.y : 0.0f;
                    x_window[i][3] = (x4.z > 0.0f) ? x4.z : 0.0f;
                    x_window[i][4] = (x4.w > 0.0f) ? x4.w : 0.0f;
                    x_window[i][5] = (right_x > 0.0f) ? right_x : 0.0f;
                } else {
                    #pragma unroll
                    for (int j = 0; j < 6; ++j) {
                        const int cur_w = tx4 + j - 1;
                        if (cur_w >= 0 && cur_w < W) {
                            float val = x[x_row_base + cur_w];
                            x_window[i][j] = (val > 0.0f) ? val : 0.0f;
                        } else x_window[i][j] = 0.0f;
                    }
                }
            }
        }

        // 3. Accumulate gradients for 4 output points
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            float g_val = gy[k];
            local_gb += g_val;
            #pragma unroll
            for (int i = 0; i < 3; ++i) {
                #pragma unroll
                for (int j = 0; j < 3; ++j) {
                    local_gw[i * 3 + j] += g_val * x_window[i][k + j];
                }
            }
        }
    }

    // Atomic add the aggregated result for this thread
    atomicAdd(&grad_biases[c], local_gb);
    #pragma unroll
    for (int i = 0; i < 9; ++i) {
        atomicAdd(&grad_weights[c * 9 + i], local_gw[i]);
    }
}

// Host-side wrapper function
at::Tensor evolution_cuda_forward(
    at::Tensor input, at::Tensor weights, at::Tensor biases) {
    
    auto output = torch::empty_like(input);
    const int B = input.size(0);
    const int C = input.size(1);
    const int H = input.size(2);
    const int W = input.size(3);

    dim3 threads(8, 32); 
    dim3 blocks(
        B * C,
        (H + threads.y - 1) / threads.y,
        (W + (threads.x * 4) - 1) / (threads.x * 4)
    );

    evolution_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weights.data_ptr<float>(), 
        biases.data_ptr<float>(), output.data_ptr<float>(),
        B, C, H, W);

    return output;
}

std::vector<at::Tensor> evolution_cuda_backward(
    at::Tensor grad_y, at::Tensor x, at::Tensor weights) {

    auto grad_x = torch::empty_like(x);
    auto grad_weights = torch::zeros_like(weights);
    auto grad_biases = torch::zeros({weights.size(0)}, weights.options());

    const int B = x.size(0);
    const int C = x.size(1);
    const int H = x.size(2);
    const int W = x.size(3);

    dim3 threads(8, 32);
    dim3 blocks(
        B * C,
        (H + threads.y - 1) / threads.y,
        (W + (threads.x * 4) - 1) / (threads.x * 4)
    );

    // 1. Compute grad_x
    evolution_backward_input_kernel<<<blocks, threads>>>(
        grad_y.data_ptr<float>(), x.data_ptr<float>(),
        weights.data_ptr<float>(), grad_x.data_ptr<float>(),
        B, C, H, W);

    // 2. Compute grad_weights and grad_biases
    // Optimization: Use float4 vectorization and 4-pixel tiling for param reduction
    dim3 threads_p(8, 32); 
    dim3 blocks_p(
        C, // Each block handles one channel to reduce atomic contention across channels
        (H + threads_p.y - 1) / threads_p.y,
        (W + (threads_p.x * 4) - 1) / (threads_p.x * 4)
    );
    evolution_backward_param_kernel<<<blocks_p, threads_p>>>(
        grad_y.data_ptr<float>(), x.data_ptr<float>(),
        grad_weights.data_ptr<float>(), grad_biases.data_ptr<float>(),
        B, C, H, W);

    return {grad_x, grad_weights, grad_biases};
}
