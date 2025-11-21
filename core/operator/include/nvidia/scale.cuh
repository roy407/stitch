#pragma once

#include <cuda_runtime.h>
#include <cstdint>

extern "C" void launch_scale_1_2_kernel(
    const uint8_t* input_y, const uint8_t* input_uv,
    int input_linesize_y, int input_linesize_uv,
    uint8_t* output_y, uint8_t* output_uv,
    int output_linesize_y, int output_linesize_uv,
    int src_w, int src_h, cudaStream_t stream);