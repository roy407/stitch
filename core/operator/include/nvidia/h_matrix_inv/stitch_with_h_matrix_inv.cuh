#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include "h_matrix_inv.h"

extern "C"
void launch_stitch_kernel_with_h_matrix_inv(uint8_t** inputs_y, uint8_t** inputs_uv, 
    int* input_linesize_y, int* input_linesize_uv, 
    uint8_t* output_y, uint8_t* output_uv,
    int output_linesize_y, int output_linesize_uv,
    int cam_num, int single_width, int width, int height, float* h_matrix_inv, float** cam_polygons, cudaStream_t stream);

#define LAUNCH_STITCH_KERNEL_WITH_H_MATRIX_INV