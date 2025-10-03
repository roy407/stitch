
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


// 双线性插值采样
__device__ uint8_t bilinear_interp(
    uint8_t* image, int stride, 
    float x, float y, int width, int height) 
{
    int x0 = static_cast<int>(x);
    int y0 = static_cast<int>(y);
    float dx = x - x0;
    float dy = y - y0;

    // 边界检查
    x0 = max(0, min(width - 1, x0));
    y0 = max(0, min(height - 1, y0));
    int x1 = min(width - 1, x0 + 1);
    int y1 = min(height - 1, y0 + 1);

    // 采样四个点
    uint8_t p00 = image[y0 * stride + x0];
    uint8_t p01 = image[y0 * stride + x1];
    uint8_t p10 = image[y1 * stride + x0];
    uint8_t p11 = image[y1 * stride + x1];

    // 插值计算
    float val = p00 * (1 - dx) * (1 - dy) + 
                p01 * dx * (1 - dy) + 
                p10 * (1 - dx) * dy + 
                p11 * dx * dy;

    return static_cast<uint8_t>(val);
}

// 计算混合权重（线性渐变）
__device__ float compute_blend_weight(int x, int y, 
    float blend_start, float blend_end) 
{
    float t = (x - blend_start) / (blend_end - blend_start);
    return fmaxf(0.0f, fminf(1.0f, t)); // 限制在[0,1]
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


__shared__ float cam_polygons[5][8]; // 每个摄像头4个顶点(x,y)
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        // Cam0 (示例：左下角区域)
        cam_polygons[0][0] = 0.0f;    cam_polygons[0][1] = 0.0f;
        cam_polygons[0][2] = 3840.0f; cam_polygons[0][3] = 0.0f;
        cam_polygons[0][4] = 3840.0f; cam_polygons[0][5] = 2160.0f;
        cam_polygons[0][6] = 0.0f;    cam_polygons[0][7] = 2160.0f;
        
        cam_polygons[1][0] = 3840.0f;    cam_polygons[1][1] = 0.0f;
        cam_polygons[1][2] = 6900.0f; cam_polygons[1][3] = 0.0f;
        cam_polygons[1][4] = 6900.0f; cam_polygons[1][5] = 2160.0f;
        cam_polygons[1][6] = 3840.0f;    cam_polygons[1][7] = 2160.0f;

        cam_polygons[2][0] = 6900.0f;    cam_polygons[2][1] = 0.0f;
        cam_polygons[2][2] = 10300.0f; cam_polygons[2][3] = 0.0f;
        cam_polygons[2][4] = 10300.0f; cam_polygons[2][5] = 2160.0f;
        cam_polygons[2][6] = 6900.0f;    cam_polygons[2][7] = 2160.0f;

        cam_polygons[3][0] = 10300.0f;    cam_polygons[3][1] = 0.0f;
        cam_polygons[3][2] = 13496.0f; cam_polygons[3][3] = 0.0f;
        cam_polygons[3][4] = 13496.0f; cam_polygons[3][5] = 2160.0f;
        cam_polygons[3][6] = 10300.0f;    cam_polygons[3][7] = 2160.0f;

        cam_polygons[4][0] = 13496.0f;    cam_polygons[4][1] = 0.0f;
        cam_polygons[4][2] = 16251.0f; cam_polygons[4][3] = 0.0f;
        cam_polygons[4][4] = 16251.0f; cam_polygons[4][5] = 2160.0f;
        cam_polygons[4][6] = 13496.0f;    cam_polygons[4][7] = 2160.0f;

    }
    __syncthreads();

    int active_cam = -1;
    for (int cam = cam_num-1; cam >= 0; --cam) {
        float* quad = cam_polygons[cam];
        if (is_point_in_quadrilateral(x, y, 
            quad[0], quad[1], quad[2], quad[3], 
            quad[4], quad[5], quad[6], quad[7])) {
            active_cam = cam;
            break;
        }
    }
    if (active_cam == -1) return;

    // --- 3. 单应性变换和采样 ---
    float* H_inv = &h_matrices[active_cam * 9];
    float src_x, src_y;
    applyHomography(H_inv, x, y, &src_x, &src_y);


    // 处理 Y 通道
    if (src_x >= 0 && src_x < single_width && src_y >= 0 && src_y < height) {
        int src_x_int = __float2int_rn(src_x);
        int src_y_int = __float2int_rn(src_y);
        output_y[y * output_linesize_y + x] = 
            inputs_y[active_cam][src_y_int * input_linesize_y[active_cam] + src_x_int];
    }

    // 处理 UV 通道（仅偶数线程处理）
    if ((x % 2 == 0) && (y % 2 == 0)) {
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
void launch_stitch_kernel_with_h_matrix(
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

