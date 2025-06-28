
#include "stitch.cuh"
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <iostream>

// NV12 stitch

__global__ void stitch_y_uv_kernel(uint8_t* const* inputs_y, uint8_t* const* inputs_uv,
                                   uint8_t* output_y, uint8_t* output_uv,
                                   int cam_num, int single_width, int width, int height) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < 640; j++) {
            int input = i * single_width + j;
            int output = i * width + thread_id * single_width + j;
            output_y[output] = inputs_y[thread_id][input];
        }
    }

    for (int i = 0; i < height / 2; i++) {
        for (int j = 0; j < 640; j++) {
            int input = i * single_width + j;
            int output = i * width + thread_id * single_width + j;
            output_uv[output] = inputs_uv[thread_id][input];
        }
    }

}

__global__ void stitch_y_uv_with_linesize_kernel_old(
    uint8_t* const* inputs_y, uint8_t* const* inputs_uv,
    int* input_linesize_y, int* input_linesize_uv,
    uint8_t* output_y, uint8_t* output_uv,
    int output_linesize_y, int output_linesize_uv,
    int cam_num, int single_width, int width, int height) {
    
    int cam_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (cam_idx >= cam_num) return;

    // 拼接 Y 分量
    for (int i = 0; i < height; i++) {
        uint8_t* input_line = inputs_y[cam_idx] + i * input_linesize_y[cam_idx];
        uint8_t* output_line = output_y + i * output_linesize_y + cam_idx * single_width;
        memcpy(output_line, input_line, single_width);
    }

    // 拼接 UV 分量（高度是 height / 2）
    for (int i = 0; i < height / 2; i++) {
        uint8_t* input_line = inputs_uv[cam_idx] + i * input_linesize_uv[cam_idx];
        uint8_t* output_line = output_uv + i * output_linesize_uv + cam_idx * single_width;
        memcpy(output_line, input_line, single_width);
    }
}

__global__ void stitch_rgb_kernel(uint8_t* const* inputs_r, uint8_t* const* inputs_g, uint8_t* const* inputs_b,
                                   uint8_t** output, int cam_num, int single_width, int width, int height) {

}

__global__ void stitch_y_uv_with_linesize_kernel(
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

__global__ void stitch_y_uv_with_linesize_and_crop_kernel(
    uint8_t* const* inputs_y, uint8_t* const* inputs_uv,
    int* input_linesize_y, int* input_linesize_uv,
    uint8_t* output_y, uint8_t* output_uv,
    int output_linesize_y, int output_linesize_uv,
    int cam_num, int single_width, int width, int height,
    int* crop)  
{
    int cam_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if(cam_idx >= cam_num || row >= height) return;

    int sum=0;
    for(int i=0;i<cam_idx;i++){
        sum +=crop[i];
    }

    int cropped_width = single_width - crop[cam_idx];
    int output_offset = cam_idx * single_width - sum;
    

    uint8_t* input_line = inputs_y[cam_idx] + row * input_linesize_y[cam_idx];
    uint8_t* output_line = output_y + row * output_linesize_y + output_offset;

    for(int i = crop[cam_idx]; i < single_width; i++) {
        output_line[i-crop[cam_idx]] = input_line[i];
    }

    if(row < height/2) {
        uint8_t* uv_input = inputs_uv[cam_idx] + row * input_linesize_uv[cam_idx];
        uint8_t* uv_output = output_uv + row * output_linesize_uv + output_offset;
        
        for(int i = crop[cam_idx]; i < single_width; i++) {
            uv_output[i - crop[cam_idx]] = uv_input[i];
        }
    }
}

extern "C"
void launch_stitch_kernel_with_crop(
    uint8_t** inputs_y, uint8_t** inputs_uv,
    int* input_linesize_y, int* input_linesize_uv,
    uint8_t* output_y, uint8_t* output_uv,
    int output_linesize_y, int output_linesize_uv,
    int cam_num, int single_width, int width, int height,
    cudaStream_t stream, int* crop)
{
    dim3 block(16, 16);
    dim3 grid((cam_num + block.x - 1) / block.x, 
              (height + block.y - 1) / block.y);

    stitch_y_uv_with_linesize_and_crop_kernel<<<grid, block, 0, stream>>>(
        inputs_y, inputs_uv,
        input_linesize_y, input_linesize_uv,
        output_y, output_uv,
        output_linesize_y, output_linesize_uv,
        cam_num, single_width, width, height,
        crop); 
}

extern "C"
void launch_stitch_kernel_raw(
    uint8_t** inputs_y, uint8_t** inputs_uv,
    int* input_linesize_y, int* input_linesize_uv,
    uint8_t* output_y, uint8_t* output_uv,
    int output_linesize_y, int output_linesize_uv,
    int cam_num, int single_width, int width, int height,
    cudaStream_t stream) {
    
    int max_threads_per_block;
    cudaDeviceGetAttribute(&max_threads_per_block, cudaDevAttrMaxThreadsPerBlock, 0);
    
    dim3 block(16, max_threads_per_block / 16); 
    dim3 grid(
        (cam_num + block.x - 1) / block.x,
        (height + block.y - 1) / block.y
    );

    stitch_y_uv_with_linesize_kernel<<<grid, block, 0, stream>>>(
        inputs_y, inputs_uv,
        input_linesize_y, input_linesize_uv,
        output_y, output_uv,
        output_linesize_y, output_linesize_uv,
        cam_num, single_width, width, height
    );
}

extern "C"
void launch_scale_1_2_kernel(uint8_t** inputs_y, uint8_t** inputs_uv,
                          int* input_linesize_y, int* input_linesize_uv,
                          int width, int height, int cam_num,
                          cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((cam_num + block.x - 1) / block.x, 
              (height + block.y - 1) / block.y);

    stitch_y_uv_with_linesize_and_crop_kernel<<<grid, block, 0, stream>>>(
        inputs_y, inputs_uv, input_linesize_y, input_linesize_uv,
        output_y, output_uv, output_linesize_y, output_linesize_uv,
        cam_num, single_width, width, height, crop); 
}

