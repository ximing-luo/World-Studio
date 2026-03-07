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

__global__ void evolution_kernel_pref(
    const float* __restrict__ x, 
    const float* __restrict__ weights, 
    const float* __restrict__ biases, 
    float* __restrict__ y,
    int B, int C, int H, int W) {

    // Index mapping: one thread handles 4 horizontal pixels along the W dimension.
    const int b_c = blockIdx.z; // (Batch * Channel) index
    const int c = b_c % C;      // Channel index for weight selection

    const int ty = blockIdx.y * blockDim.y + threadIdx.y;
    const int tx4 = (blockIdx.x * blockDim.x + threadIdx.x) * 4;

    // Boundary check for H and W
    if (ty >= H || tx4 >= W) {
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

// Host-side wrapper function
at::Tensor evolution_cuda_forward_pref(
    at::Tensor input, at::Tensor weights, at::Tensor biases) {
    
    auto output = torch::empty_like(input);
    const int B = input.size(0);
    const int C = input.size(1);
    const int H = input.size(2);
    const int W = input.size(3);

    // Ensure W is a multiple of 4 for optimal float4 usage, 
    // though the kernel handles boundary cases.
    dim3 threads(8, 32); 
    dim3 blocks(
        (W + (threads.x * 4) - 1) / (threads.x * 4),
        (H + threads.y - 1) / threads.y,
        B * C
    );

    evolution_kernel_pref<<<blocks, threads>>>(
        input.data_ptr<float>(), weights.data_ptr<float>(), 
        biases.data_ptr<float>(), output.data_ptr<float>(),
        B, C, H, W);

    return output;
}
