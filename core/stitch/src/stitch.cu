
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

__device__ bool is_point_in_quadrilateral(float x, float y,
    float x1, float y1, float x2, float y2,
    float x3, float y3, float x4, float y4)
{
    // 向量叉积法判断点是否在凸四边形内
    auto cross = [](float ax, float ay, float bx, float by) {
        return ax * by - ay * bx;
    };

    float d1 = cross(x - x1, y - y1, x2 - x1, y2 - y1);
    float d2 = cross(x - x2, y - y2, x3 - x2, y3 - y2);
    float d3 = cross(x - x3, y - y3, x4 - x3, y4 - y3);
    float d4 = cross(x - x4, y - y4, x1 - x4, y1 - y4);

    return (d1 * d3 >= 0) && (d2 * d4 >= 0);
}

__global__ void USE_HNI(
    uint8_t* const* inputs_y, uint8_t* const* inputs_uv,
    int* input_linesize_y, int* input_linesize_uv,
    float* h_matrices, uint8_t* output_y, uint8_t* output_uv,
    int output_linesize_y, int output_linesize_uv,
    int cam_num, int single_width, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;
    // 摄像头1的四边形顶点
    bool in_cam1 = is_point_in_quadrilateral(x, y, 
        0.0f, 12.0f,    
        640.0f, 12.0f,  
        640.0f, 372.0f, 
        0.0f, 372.0f); 
    // bool in_cam1 = is_point_in_quadrilateral(x, y, 
    //     0.0f, 0.0f,    
    //     640.0f, 0.0f,  
    //     640.0f, 360.0f, 
    //     0.0f, 360.0f); 

    // 摄像头2的四边形顶点
    bool in_cam2 = false;
    if (!in_cam1) {
        in_cam2 = is_point_in_quadrilateral(x, y,
            568.38f, 16.62f,   
            1237.14f, 0.42f,   
            1250.31f, 369.77f, 
            580.55f, 375.24f); 
    }
    //     if (!in_cam1) {
    //     in_cam2 = is_point_in_quadrilateral(x, y,
    //         568.0f, 0.0f,   
    //         1237.0f, 0.0f,   
    //         1250.0f, 360.0f, 
    //         580.0f, 360.0f); 
    // }



    if (!in_cam1 && !in_cam2) return;

    int active_cam = in_cam2 ? 1 : 0;
    float* H_inv = &h_matrices[active_cam * 9];

    float src_x, src_y;
    applyHomography(H_inv, x, y, &src_x, &src_y);

    if (src_x >= 0 && src_x < single_width && src_y >= 0 && src_y < height) {
        int src_x_int = __float2int_rn(src_x);
        int src_y_int = __float2int_rn(src_y);
        output_y[y * output_linesize_y + x] = 
            inputs_y[active_cam][src_y_int * input_linesize_y[active_cam] + src_x_int];
    }


    if ((threadIdx.x % 2 == 0) && (threadIdx.y % 2 == 0) &&
        (x % 2 == 0) && (y % 2 == 0)) 
    {
        float uv_x = src_x / 2.0f;
        float uv_y = src_y / 2.0f;
        
        if (uv_x >= 0 && uv_x < single_width/2 && uv_y >= 0 && uv_y < height/2) {
            int src_uv_x = __float2int_rn(uv_x);
            int src_uv_y = __float2int_rn(uv_y);
            int uv_pitch = input_linesize_uv[active_cam];
            
            int out_uv_idx = (y/2) * output_linesize_uv + (x/2) * 2;
            output_uv[out_uv_idx] = inputs_uv[active_cam][src_uv_y * uv_pitch + src_uv_x * 2];
            output_uv[out_uv_idx + 1] = inputs_uv[active_cam][src_uv_y * uv_pitch + src_uv_x * 2 + 1];
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
        (width + block.x - 1) / block.x,  // 水平方向块数
        (height + block.y - 1) / block.y,        // 垂直方向块数
        cam_num                                  // 摄像头数量
    );

    USE_HNI<<<grid, block, 0, stream>>>(
        inputs_y, inputs_uv,
        input_linesize_y, input_linesize_uv,h_matrices,
        output_y, output_uv,
        output_linesize_y, output_linesize_uv,
        cam_num, single_width, width, height
    );
}



