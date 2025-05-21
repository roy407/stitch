#pragma once

#include <cuda_runtime.h>
#include <cstdint>

// 调用接口
extern "C" void launch_stitch_kernel(uint8_t* inputs[5], uint8_t* output, int width, int height, cudaStream_t stream);