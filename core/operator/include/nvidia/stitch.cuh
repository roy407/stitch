#pragma once

#include <cuda_runtime.h>
#include <cstdint>


extern "C"

void launch_stitch_kernel_with_crop(uint8_t** inputs_y, uint8_t** inputs_uv,
                            int* input_linesize_y, int* input_linesize_uv,
                            uint8_t* output_y, uint8_t* output_uv,
                            int output_linesize_y, int output_linesize_uv,
                            int cam_num, int single_width, int width, int height,
                            cudaStream_t stream,int* crop);

extern "C"
void launch_stitch_kernel_raw(uint8_t** inputs_y, uint8_t** inputs_uv,
                            int* input_linesize_y, int* input_linesize_uv,
                            uint8_t* output_y, uint8_t* output_uv,
                            int output_linesize_y, int output_linesize_uv,
                            int cam_num, int single_width, int width, int height,
                            cudaStream_t stream);

extern "C"
void launch_stitch_kernel_with_h_matrix(uint8_t** inputs_y, uint8_t** inputs_uv, 
                            int* input_linesize_y, int* input_linesize_uv, 
                            float* h_matrix, float** cam_polygons, uint8_t* output_y, 
                            uint8_t* output_uv, int output_linesize_y, 
                            int output_linesize_uv, int cam_num, 
                            int single_width, int width, int height, 
                            cudaStream_t stream);

extern "C"
void launch_stitch_kernel_with_mapping_table(
    uint8_t** inputs_y, uint8_t** inputs_uv,
    int* input_linesize_y, int* input_linesize_uv,
    uint8_t* output_y, uint8_t* output_uv,
    int output_linesize_y, int output_linesize_uv,
    int cam_num, int single_width, int width, int height, const uint16_t* mapping_table,
    cudaStream_t stream);
