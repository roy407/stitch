
#include "stitch.cuh"
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <iostream>


__global__ void stitch_y_uv_kernel(uint8_t* const* inputs_y, uint8_t* const* inputs_uv,
                                   uint8_t* output_y, uint8_t* output_uv,
                                   int cam_num, int single_width, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;


    int cam_idx = x / single_width;
    int local_x = x % single_width;
    // === Y 平面 ===
    int input_idx = y * single_width + local_x;
    int output_idx = y * width + x;
    
    output_y[output_idx] = inputs_y[cam_idx][input_idx];

    if (y < height / 2) {
    int input_idx = y * single_width + local_x;
    int output_idx = y * width + x;
    
    output_uv[output_idx] = inputs_uv[cam_idx][input_idx];
    }
}

extern "C"
void launch_stitch_y_uv_kernel(uint8_t** inputs_y, uint8_t** inputs_uv,
                               uint8_t* output_y, uint8_t* output_uv,
                               int cam_num, int single_width, int width, int height,
                               cudaStream_t stream) {
    dim3 block(32, 16);
    dim3 grid(100, 40);

    stitch_y_uv_kernel<<<grid, block, 0, stream>>>(inputs_y, inputs_uv, output_y, output_uv,
                                                   cam_num, 640, 640, 640);
}
