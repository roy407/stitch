#include "resize.cuh"
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <iostream>

__global__ void ReSizeKernel_Bilinear_NV12(
    const uint8_t* pInYData, const uint8_t* pInUVData, 
    int pInWidth, int pInHeight, int pInYStride, int pInUVStride,
    uint8_t*  pOutYData, 
    uint8_t*  pOutUVData, 
    int pOutWidth, int pOutHeight, int pOutYStride, int pOutUVStride)
{
    int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    int tidy = blockIdx.y * blockDim.y + threadIdx.y;

    if (tidx >= pOutWidth || tidy >= pOutHeight) return;

    float srcX = (tidx + 0.5f) * pInWidth / pOutWidth - 0.5f;
    float srcY = (tidy + 0.5f) * pInHeight / pOutHeight - 0.5f;

    srcX = fmaxf(0.0f, fminf(srcX, pInWidth - 1.0f));
    srcY = fmaxf(0.0f, fminf(srcY, pInHeight - 1.0f));

    int x0 = (int)srcX;
    int y0 = (int)srcY;
    int x1 = min(x0 + 1, pInWidth - 1);
    int y1 = min(y0 + 1, pInHeight - 1);

    float dx = srcX - x0;
    float dy = srcY - y0;


    float w00 = (1.0f - dx) * (1.0f - dy);
    float w10 = dx * (1.0f - dy);
    float w01 = (1.0f - dx) * dy;
    float w11 = dx * dy;

    int in_y00 = y0 * pInYStride + x0;
    int in_y10 = y1 * pInYStride + x0;
    int in_y01 = y0 * pInYStride + x1;
    int in_y11 = y1 * pInYStride + x1;

    int out_y = tidy * pOutYStride + tidx;

    float y_val = 
        pInYData[in_y00] * w00 +
        pInYData[in_y10] * w10 +
        pInYData[in_y01] * w01 +
        pInYData[in_y11] * w11;

    pOutYData[out_y] = (uint8_t)fminf(fmaxf(y_val, 0.0f), 255.0f);

  
    if ((tidx % 2 == 0) && (tidy % 2 == 0)) {
        int uv_x = tidx / 2;
        int uv_y = tidy / 2;
        
        if (uv_x < pOutWidth / 2 && uv_y < pOutHeight / 2) {
         
            float src_uv_x = (uv_x + 0.5f) * (pInWidth / 2) / (pOutWidth / 2) - 0.5f;
            float src_uv_y = (uv_y + 0.5f) * (pInHeight / 2) / (pOutHeight / 2) - 0.5f;

            src_uv_x = fmaxf(0.0f, fminf(src_uv_x, pInWidth / 2 - 1.0f));
            src_uv_y = fmaxf(0.0f, fminf(src_uv_y, pInHeight / 2 - 1.0f));

            int uv_x0 = (int)src_uv_x;
            int uv_y0 = (int)src_uv_y;
            int uv_x1 = min(uv_x0 + 1, pInWidth / 2 - 1);
            int uv_y1 = min(uv_y0 + 1, pInHeight / 2 - 1);

            float uv_dx = src_uv_x - uv_x0;
            float uv_dy = src_uv_y - uv_y0;

            float uv_w00 = (1.0f - uv_dx) * (1.0f - uv_dy);
            float uv_w10 = uv_dx * (1.0f - uv_dy);
            float uv_w01 = (1.0f - uv_dx) * uv_dy;
            float uv_w11 = uv_dx * uv_dy;


            int in_uv00 = uv_y0 * pInUVStride + uv_x0 * 2;  
            int in_uv10 = uv_y1 * pInUVStride + uv_x0 * 2;
            int in_uv01 = uv_y0 * pInUVStride + uv_x1 * 2;
            int in_uv11 = uv_y1 * pInUVStride + uv_x1 * 2;

            int out_uv = uv_y * pOutUVStride + uv_x * 2;

            // U分量（偶数位置）
            float u_val = 
                pInUVData[in_uv00] * uv_w00 +
                pInUVData[in_uv10] * uv_w10 +
                pInUVData[in_uv01] * uv_w01 +
                pInUVData[in_uv11] * uv_w11;

            pOutUVData[out_uv] = (uint8_t)fminf(fmaxf(u_val, 0.0f), 255.0f);

            // V分量（奇数位置）
            float v_val = 
                pInUVData[in_uv00 + 1] * uv_w00 +
                pInUVData[in_uv10 + 1] * uv_w10 +
                pInUVData[in_uv01 + 1] * uv_w01 +
                pInUVData[in_uv11 + 1] * uv_w11;

            pOutUVData[out_uv + 1] = (uint8_t)fminf(fmaxf(v_val, 0.0f), 255.0f);
        }
    }
}

extern "C" 
void ReSize(
    const uint8_t* pInYData, const uint8_t* pInUVData, 
    int pInWidth, int pInHeight, int pInYStride, int pInUVStride,
    uint8_t* pOutYData, uint8_t* pOutUVData, 
    int pOutWidth, int pOutHeight, int pOutYStride, int pOutUVStride,
    cudaStream_t stream)
{
    dim3 block(16, 16);
    dim3 grid(
        (pOutWidth + block.x - 1) / block.x,
        (pOutHeight + block.y - 1) / block.y
    );

    ReSizeKernel_Bilinear_NV12<<<grid, block, 0, stream>>>(
        pInYData, pInUVData, 
        pInWidth, pInHeight, pInYStride, pInUVStride,
        pOutYData, pOutUVData, 
        pOutWidth, pOutHeight, pOutYStride, pOutUVStride
    );

}