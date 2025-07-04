
#include "stitch.cuh"
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <iostream>
#include <device_launch_parameters.h>
// // NV12 stitch

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

__device__ void applyHomography(float* H, float x, float y, float* out_x, float* out_y) {
    float denominator = H[6]*x + H[7]*y + H[8];
    if (fabsf(denominator) < 1e-6f) {
        *out_x = -1;
        *out_y = -1;
        return;
    }
    *out_x = (H[0]*x + H[1]*y + H[2]) / denominator;
    *out_y = (H[3]*x + H[4]*y + H[5]) / denominator;
}

__device__ uint8_t bilinearInterpolation(uint8_t* image, int width, int height, int pitch, 
                                       float x, float y, int channel) {
    // 边界保护
    x = fminf(fmaxf(x, 0.0f), width - 1.001f);
    y = fminf(fmaxf(y, 0.0f), height - 1.001f);

    int x1 = __float2int_rd(x);
    int y1 = __float2int_rd(y);
    int x2 = min(x1 + 1, width - 1);
    int y2 = min(y1 + 1, height - 1);

    float dx = x - x1;
    float dy = y - y1;

    uint8_t p11 = image[y1 * pitch + x1 + channel];
    uint8_t p12 = image[y1 * pitch + x2 + channel];
    uint8_t p21 = image[y2 * pitch + x1 + channel];
    uint8_t p22 = image[y2 * pitch + x2 + channel];

    return __float2uint_rn((1-dx)*(1-dy)*p11 + dx*(1-dy)*p12 + 
                         (1-dx)*dy*p21 + dx*dy*p22);
}

__device__ uint8_t bilinearInterpolationUV(uint8_t* uv_plane, int width, int height, int pitch,
                                         float x, float y, int channel) {
    // 边界保护
    x = fminf(fmaxf(x, 0.0f), width - 1.001f);
    y = fminf(fmaxf(y, 0.0f), height - 1.001f);

    int x1 = __float2int_rd(x);
    int y1 = __float2int_rd(y);
    int x2 = min(x1 + 1, width - 1);
    int y2 = min(y1 + 1, height - 1);

    float dx = x - x1;
    float dy = y - y1;

    // NV12格式：UV交错存储
    uint8_t p11 = uv_plane[y1 * pitch + x1 * 2 + channel];
    uint8_t p12 = uv_plane[y1 * pitch + x2 * 2 + channel];
    uint8_t p21 = uv_plane[y2 * pitch + x1 * 2 + channel];
    uint8_t p22 = uv_plane[y2 * pitch + x2 * 2 + channel];

    return __float2uint_rn((1-dx)*(1-dy)*p11 + dx*(1-dy)*p12 + 
                         (1-dx)*dy*p21 + dx*dy*p22);
}

__global__ void stitch_withH(
    uint8_t* const* inputs_y, uint8_t* const* inputs_uv,
    int* input_linesize_y, int* input_linesize_uv,
    float* h_matrices, uint8_t* output_y, uint8_t* output_uv,
    int output_linesize_y, int output_linesize_uv,
    int cam_num, int single_width, int width, int height)
{
    int cam_idx = blockIdx.z;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;


    if (x >= single_width || y >= height || cam_idx >= cam_num) return;

    float* H = &h_matrices[cam_idx * 9];

    float out_x, out_y;
    applyHomography(H, x, y, &out_x, &out_y);


    if (out_x >= 0 && out_x < width && out_y >= 0 && out_y < height) {

        int out_x_int = __float2int_rn(out_x);
        int out_y_int = __float2int_rn(out_y);
        output_y[out_y_int * output_linesize_y + out_x_int] = inputs_y[cam_idx][y * input_linesize_y[cam_idx] + x];

        if (x % 2 == 0 && y % 2 == 0) {
            float uv_x = x / 2;  
            float uv_y = y / 2;
            int out_uv_x = out_x_int / 2;
            int out_uv_y = out_y_int / 2;

            if (out_uv_x >= 0 && out_uv_x < width/2 && out_uv_y >= 0 && out_uv_y < height/2) {

            int src_uv_x = __float2int_rn(uv_x);  
            int src_uv_y = __float2int_rn(uv_y);
    
            int uv_pitch = input_linesize_uv[cam_idx];
            uint8_t u_val = inputs_uv[cam_idx][src_uv_y * uv_pitch + src_uv_x * 2];      
            uint8_t v_val = inputs_uv[cam_idx][src_uv_y * uv_pitch + src_uv_x * 2 + 1];  
    
            int uv_index = out_uv_y * output_linesize_uv + out_uv_x * 2;
            output_uv[uv_index] = u_val;
            output_uv[uv_index + 1] = v_val;
            }
        }
    }
}


extern "C"
void launch_stitch_kernel(
    uint8_t** inputs_y, uint8_t** inputs_uv,
    int* input_linesize_y, int* input_linesize_uv,float* h_matrices,
    uint8_t* output_y, uint8_t* output_uv,
    int output_linesize_y, int output_linesize_uv,
    int cam_num, int single_width, int width, int height,
    cudaStream_t stream) {
    
    int max_threads_per_block;
    cudaDeviceGetAttribute(&max_threads_per_block, cudaDevAttrMaxThreadsPerBlock, 0);
    
    dim3 block(16,16); 
    dim3 grid(
        (single_width + block.x - 1) / block.x,  // 水平方向块数
        (height + block.y - 1) / block.y,        // 垂直方向块数
        cam_num                                  // 摄像头数量
    );

    stitch_withH<<<grid, block, 0, stream>>>(
        inputs_y, inputs_uv,
        input_linesize_y, input_linesize_uv,h_matrices,
        output_y, output_uv,
        output_linesize_y, output_linesize_uv,
        cam_num, single_width, width, height
    );
}



