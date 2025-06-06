#pragma once

#include <cuda_runtime.h>
#include <cstdint>

// #define NV12_STITCH
// #define RGB24_STITCH

extern "C"
// void launch_stitch_kernel(uint8_t** inputs_y, uint8_t** inputs_uv,
//                           int* input_linesize_y, int* input_linesize_uv,
//                           uint8_t* output_y, uint8_t* output_uv,
//                           int output_linesize_y, int output_linesize_uv,
//                           int cam_num, int single_width, int width, int height,
//                           cudaStream_t stream,int* crop_pixels);


void launch_stitch_kernel(uint8_t** inputs_y, uint8_t** inputs_uv,
                          int* input_linesize_y, int* input_linesize_uv,
                          uint8_t* output_y, uint8_t* output_uv,
                          int output_linesize_y, int output_linesize_uv,
                          int cam_num, int single_width, int width, int height,
                          cudaStream_t stream,int* crop);