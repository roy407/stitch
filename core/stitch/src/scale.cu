#include "scale.cuh"
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <iostream>

// NV12 scale

extern "C" __global__ __aicore__ void scale_1_2_y_uv_with_nomem_kernel(uint8_t** inputs_y, uint8_t** inputs_uv,
                          int* input_linesize_y, int* input_linesize_uv,
                          int pre_width, int pre_height, int cam_num) {
    int cam_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int out_width = pre_width / 2;
    int out_height = pre_height / 2;
    if(cam_idx >= cam_num || row >= out_height) return;
    uint8_t* input_y1 = inputs_y[cam_idx] + row * 2 * input_linesize_y[cam_idx];
    uint8_t* input_y2 = inputs_y[cam_idx] + (row * 2 + 1)* input_linesize_y[cam_idx];
    uint8_t* output_y = inputs_y[cam_idx] + row * input_linesize_y[cam_idx];
    for(int i = 0; i < out_width; i++) {
        output_y[i] = (input_y1[i * 2] + input_y1[i * 2 + 1] + input_y2[i * 2] + input_y2[i * 2 + 1]) / 4;
    }

    // u v u v u v u v
    // 0 1 2 3 4 5 6 7
    // 0 0     1 1

    if(row < out_height / 2) {
        uint8_t* input_uv1 = inputs_uv[cam_idx] + (row * 2) * input_linesize_uv[cam_idx];
        uint8_t* input_uv2 = inputs_uv[cam_idx] + (row * 2 + 1) * input_linesize_uv[cam_idx];
        uint8_t* output_uv = inputs_uv[cam_idx] + row * input_linesize_uv[cam_idx];

        for(int i = 0; i < out_width/2; i++) {
            const int src_idx = i * 4;
            const int dst_idx = i * 2;
            uint8_t u = (input_uv1[src_idx]     + input_uv1[src_idx + 2] +
                         input_uv2[src_idx]     + input_uv2[src_idx + 2]) / 4;
            uint8_t v = (input_uv1[src_idx + 1] + input_uv1[src_idx + 3] +
                         input_uv2[src_idx + 1] + input_uv2[src_idx + 3]) / 4;
            output_uv[dst_idx] = u;
            output_uv[dst_idx + 1] = v;
        }
    }
}

extern "C"
void launch_scale_1_2_kernel(uint8_t** inputs_y, uint8_t** inputs_uv,
                          int* input_linesize_y, int* input_linesize_uv,
                          int pre_width, int pre_height, int cam_num,
                          cudaStream_t stream) {
    dim3 block(8, 32);
    dim3 grid((cam_num + block.x - 1) / block.x, 
              (pre_height / 2 + block.y - 1) / block.y);

    scale_1_2_y_uv_with_nomem_kernel<<<grid, block, 0, stream>>>(
        inputs_y, inputs_uv, input_linesize_y, input_linesize_uv,
        pre_width, pre_height, cam_num); 
}