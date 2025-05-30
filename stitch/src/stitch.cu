
#include "stitch.cuh"
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <iostream>


__global__ void stitch_y_uv_kernel(uint8_t* const* inputs_y, uint8_t* const* inputs_uv,
                                   uint8_t* output_y, uint8_t* output_uv,
                                   int cam_num, int single_width, int width, int height) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int num_threads = blockDim.x * gridDim.x;

    // === Y 平面 ===
    for (int y = thread_id; y < height; y += num_threads) {
        for (int cam_idx = 0; cam_idx < cam_num; ++cam_idx) {
            for (int local_x = 0; local_x < single_width; ++local_x) {
                int global_x = cam_idx * single_width + local_x;
                int input_idx = y * single_width + local_x;
                int output_idx = y * width + global_x;
                output_y[output_idx] = inputs_y[cam_idx][input_idx];
            }
        }
    }

    // === UV 平面 (NV12: interleaved, height / 2) ===
    for (int y = thread_id; y < height / 2; y += num_threads) {
        for (int cam_idx = 0; cam_idx < cam_num; ++cam_idx) {
            for (int local_x = 0; local_x < single_width; local_x += 2) {  // process UV pair
                int global_x = cam_idx * single_width + local_x;
                int input_idx = y * single_width + local_x;
                int output_idx = y * width + global_x;

                // Copy two bytes at a time: U and V
                output_uv[output_idx]     = inputs_uv[cam_idx][input_idx];
                output_uv[output_idx + 1] = inputs_uv[cam_idx][input_idx + 1];
            }
        }
    }
}


extern "C"
void launch_stitch_y_uv_kernel(uint8_t** inputs_y, uint8_t** inputs_uv,
                               uint8_t* output_y, uint8_t* output_uv,
                               int cam_num, int single_width, int width, int height,
                               cudaStream_t stream) {
    int threads = 32;  // 你可以根据需要调整
    int blocks = 1;    // 只用一个 block

    stitch_y_uv_kernel<<<blocks, threads, 0, stream>>>(inputs_y, inputs_uv, output_y, output_uv,
                                                       cam_num, 640, 640, 640);
}

