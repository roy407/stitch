
#include "stitch.cuh"
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <iostream>

// __global__ void stitch_y_uv_with_linesize_kernel(
//     uint8_t* const* inputs_y, uint8_t* const* inputs_uv,
//     int* input_linesize_y, int* input_linesize_uv,
//     uint8_t* output_y, uint8_t* output_uv,
//     int output_linesize_y, int output_linesize_uv,
//     int cam_num, int single_width, int width, int height) {
    
//     int cam_idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (cam_idx >= cam_num) return;

//     // 拼接 Y 分量
//     for (int i = 0; i < height; i++) {
//         uint8_t* input_line = inputs_y[cam_idx] + i * input_linesize_y[cam_idx];
//         uint8_t* output_line = output_y + i * output_linesize_y + cam_idx * single_width;
//         memcpy(output_line, input_line, single_width);
//     }

//     // 拼接 UV 分量（高度是 height / 2）
//     for (int i = 0; i < height / 2; i++) {
//         uint8_t* input_line = inputs_uv[cam_idx] + i * input_linesize_uv[cam_idx];
//         uint8_t* output_line = output_uv + i * output_linesize_uv + cam_idx * single_width;
//         memcpy(output_line, input_line, single_width);
//     }
// }

// extern "C"
// void launch_stitch_kernel(uint8_t** inputs_y, uint8_t** inputs_uv,
//                           int* input_linesize_y, int* input_linesize_uv,
//                           uint8_t* output_y, uint8_t* output_uv,
//                           int output_linesize_y, int output_linesize_uv,
//                           int cam_num, int single_width, int width, int height,
//                           cudaStream_t stream) {
//     int threads = cam_num;  // 每个相机一个线程
//     int blocks = 1;

//     stitch_y_uv_with_linesize_kernel<<<blocks, threads, 0, stream>>>(
//         inputs_y, inputs_uv,
//         input_linesize_y, input_linesize_uv,
//         output_y, output_uv,
//         output_linesize_y, output_linesize_uv,
//         cam_num, single_width, width, height
//     );
// }
__global__ void optimized_stitch_kernel(
    uint8_t* const* inputs_y, uint8_t* const* inputs_uv,
    int* input_linesize_y, int* input_linesize_uv,
    uint8_t* output_y, uint8_t* output_uv,
    int output_linesize_y, int output_linesize_uv,
    int cam_num, int single_width, int width, int height) {
    
    int cam_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if(cam_idx >= cam_num || row >= height) return;

    // 处理Y分量
    uint8_t* input_line = inputs_y[cam_idx] + row * input_linesize_y[cam_idx];
    uint8_t* output_line = output_y + row * output_linesize_y + cam_idx * single_width;
    for(int i=0; i<single_width; i++) {
        output_line[i] = input_line[i];
    }

    // 处理UV分量(每两行Y对应一行UV)
    if(row < height/2) {
        uint8_t* uv_input = inputs_uv[cam_idx] + row * input_linesize_uv[cam_idx];
        uint8_t* uv_output = output_uv + row * output_linesize_uv + cam_idx * single_width;
        for(int i=0; i<single_width; i++) {
            uv_output[i] = uv_input[i];
        }
    }
}

__global__ void stitch_y_uv_64(
    uint8_t* const* inputs_y, uint8_t* const* inputs_uv,
    int* input_linesize_y, int* input_linesize_uv,
    uint8_t* output_y, uint8_t* output_uv,
    int output_linesize_y, int output_linesize_uv,
    int cam_num, int single_width, int width, int height,
    int* crp)  
{
    int cam_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if(cam_idx >= cam_num || row >= height) return;

    int sum=0;
    for(int i=0;i<cam_idx;i++){
        sum +=crp[i];
    }

    int cropped_width = single_width - crp[cam_idx];
    int output_offset = cam_idx * single_width - sum;
    

    uint8_t* input_line = inputs_y[cam_idx] + row * input_linesize_y[cam_idx];
    uint8_t* output_line = output_y + row * output_linesize_y + output_offset;

    for(int i = crp[cam_idx]; i < single_width; i++) {
        output_line[i-crp[cam_idx]] = input_line[i];
    }

    if(row < height/2) {
        uint8_t* uv_input = inputs_uv[cam_idx] + row * input_linesize_uv[cam_idx];
        uint8_t* uv_output = output_uv + row * output_linesize_uv + output_offset;
        
        for(int i = crp[cam_idx]; i < single_width; i++) {
            uv_output[i - crp[cam_idx]] = uv_input[i];
        }
    }
}



extern "C"
void launch_stitch_kernel(
    uint8_t** inputs_y, uint8_t** inputs_uv,
    int* input_linesize_y, int* input_linesize_uv,
    uint8_t* output_y, uint8_t* output_uv,
    int output_linesize_y, int output_linesize_uv,
    int cam_num, int single_width, int width, int height,
    cudaStream_t stream, int* crp)  // 修改参数
{
    dim3 block(16, 16);
    dim3 grid((cam_num + block.x - 1) / block.x, 
              (height + block.y - 1) / block.y);

    stitch_y_uv_64<<<grid, block, 0, stream>>>(
        inputs_y, inputs_uv,
        input_linesize_y, input_linesize_uv,
        output_y, output_uv,
        output_linesize_y, output_linesize_uv,
        cam_num, single_width, width, height,
        crp); 
}