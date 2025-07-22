#pragma once

#include <acl/acl.h>
#include <cstdint>


extern "C"

void launch_stitch_kernel_with_crop(uint8_t** inputs_y, uint8_t** inputs_uv,
                            int* input_linesize_y, int* input_linesize_uv,
                            uint8_t* output_y, uint8_t* output_uv,
                            int output_linesize_y, int output_linesize_uv,
                            int cam_num, int single_width, int width, int height,
                            void* stream,int* crop);

extern "C"
void launch_stitch_kernel_raw(uint8_t** inputs_y, uint8_t** inputs_uv,
                            int* input_linesize_y, int* input_linesize_uv,
                            uint8_t* output_y, uint8_t* output_uv,
                            int output_linesize_y, int output_linesize_uv,
                            int cam_num, int single_width, int width, int height,
                            void* stream);

extern "C"
void launch_stitch_kernel_with_h_matrix(uint8_t** inputs_y, uint8_t** inputs_uv, 
                            int* input_linesize_y, int* input_linesize_uv, 
                            float* h_matrices, uint8_t* output_y, 
                            uint8_t* output_uv, int output_linesize_y, 
                            int output_linesize_uv, int cam_num, 
                            int single_width, int width, int height, 
                            aclrtStream stream);
