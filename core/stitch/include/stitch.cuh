// #pragma once

// #include <cuda_runtime.h>
// #include <cstdint>

// // #define NV12_STITCH
// // #define RGB24_STITCH

// extern "C"
// void launch_stitch_kernel(uint8_t** inputs_y, uint8_t** inputs_uv,
//                           int* input_linesize_y, int* input_linesize_uv,
//                           uint8_t* output_y, uint8_t* output_uv,
//                           int output_linesize_y, int output_linesize_uv,
//                           int cam_num, int single_width, int width, int height,
//                           cudaStream_t stream);




#pragma once

#include <cuda_runtime.h>
#include <cstdint>

// 透视变换拼接核函数声明
extern "C" {
    void launch_stitch_kernel(
        uint8_t** inputs_y,          // 输入Y分量指针数组(设备指针)
        uint8_t** inputs_uv,         // 输入UV分量指针数组(设备指针)
        int* input_linesize_y,       // 输入Y分量的行跨度数组
        int* input_linesize_uv,      // 输入UV分量的行跨度数组
        float* h_matrices,           // 透视变换矩阵数组(设备指针)
        uint8_t* output_y,           // 输出Y分量(设备指针)
        uint8_t* output_uv,          // 输出UV分量(设备指针)
        int output_linesize_y,       // 输出Y分量的行跨度
        int output_linesize_uv,      // 输出UV分量的行跨度
        int cam_num,                // 相机数量
        int single_width,           // 单个相机图像宽度
        int width,                  // 输出图像宽度
        int height,                 // 输出图像高度
        cudaStream_t stream         // CUDA流
    );
}

// 透视变换设备函数声明
extern "C" __device__ void applyHomography(
    float* H,       // 透视变换矩阵
    float x,        // 输入x坐标
    float y,        // 输入y坐标
    float* out_x,   // 输出x坐标
    float* out_y    // 输出y坐标
);
__device__ bool is_point_in_quadrilateral(float x, float y,
    float x1, float y1, float x2, float y2,
    float x3, float y3, float x4, float y4);
__device__ float compute_blend_weight(int x, int y, 
    float blend_start, float blend_end);
 __device__ uint8_t bilinear_interp(
    uint8_t* image, int stride, 
    float x, float y, int width, int height);

