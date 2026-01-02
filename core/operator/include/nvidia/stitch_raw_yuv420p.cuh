#pragma once

#include <cuda_runtime.h>
#include <cstdint>

extern "C"
void launch_stitch_kernel_raw_yuv420p(
                            uint8_t** inputs_y, uint8_t** inputs_u, uint8_t** inputs_v,
                            int* input_linesize_y, int* input_linesize_u, int* input_linesize_v,
                            uint8_t* output_y, uint8_t* output_uv,
                            int output_linesize_y, int output_linesize_uv,
                            int cam_num, int single_width, int width, int height,
                            cudaStream_t stream);

#define LAUNCH_STITCH_KERNEL_RAW_YUV420P