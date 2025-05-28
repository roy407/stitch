
#include "stitch.cuh"
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>


// 核函数
__global__ void stitch_kernel(uint8_t* const* inputs, uint8_t* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int single_width = width;
    int cam_idx = x / single_width;
    int local_x = x % single_width;

    int input_idx = y * single_width + local_x;
    int output_idx = y * width + x;

    output[output_idx] = inputs[cam_idx][input_idx];
}

extern "C"
void launch_stitch_kernel(uint8_t** inputs, uint8_t* output, int width, int height, cudaStream_t stream) {
    dim3 block(32, 32);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    stitch_kernel<<<grid, block, 0, stream>>>(inputs, output, width, height);
}