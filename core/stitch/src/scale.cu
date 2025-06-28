#include "scale.cuh"
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <iostream>

// NV12 scale

__global__ void scale_y_uv_with_1_2_kernel(uint8_t** inputs_y, uint8_t** inputs_uv,
                          int* input_linesize_y, int* input_linesize_uv,
                          int width, int height, int cam_num) {
    int x = 0;
}

extern "C"
void launch_scale_1_2_kernel(uint8_t** inputs_y, uint8_t** inputs_uv,
                          int* input_linesize_y, int* input_linesize_uv,
                          int width, int height, int cam_num,
                          cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((block.x - 1) / block.x, 
              (block.y - 1) / block.y);

    scale_y_uv_with_1_2_kernel<<<grid, block, 0, stream>>>(
        inputs_y, inputs_uv, input_linesize_y, input_linesize_uv,
        width, height, cam_num); 
}