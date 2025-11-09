#pragma once
#include <cuda_runtime.h>
#include <cstdint>

extern "C"
void launch_stitch_kernel_with_mapping_table(
    uint8_t** inputs_y, uint8_t** inputs_uv,
    int* input_linesize_y, int* input_linesize_uv,
    uint8_t* output_y, uint8_t* output_uv,
    int output_linesize_y, int output_linesize_uv,
    int cam_num, int single_width, int width, int height, const cudaTextureObject_t mapping_table,
    cudaStream_t stream1, cudaStream_t stream2);

#define LAUNCH_STITCH_KERNEL_WITH_MAPPING_TABLE