#pragma once

#include <cuda_runtime.h>
#include <cstdint>

extern "C"
void launch_stitch_y_uv_kernel(uint8_t** inputs_y, uint8_t** inputs_uv,
                               uint8_t* output_y, uint8_t* output_uv,
                               int cam_num, int single_width, int width, int height,
                               cudaStream_t stream);