#pragma once

#include <cuda_runtime.h>
#include <cstdint>

extern "C"
void launch_scale_1_2_kernel(uint8_t** inputs_y, uint8_t** inputs_uv,
                          int* input_linesize_y, int* input_linesize_uv,
                          int width, int height, int cam_num,
                          cudaStream_t stream);