#pragma once

#include <cuda_runtime.h>
#include <cstdint>

extern "C" void ReSize(
    const uint8_t* pInYData, const uint8_t* pInUVData,
    int pInWidth, int pInHeight, int pInYStride, int pInUVStride,
    uint8_t* pOutYData, uint8_t* pOutUVData, 
    int pOutWidth, int pOutHeight, int pOutYStride, int pOutUVStride,
    cudaStream_t stream);