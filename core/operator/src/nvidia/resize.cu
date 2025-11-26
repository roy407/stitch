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

    // 经典的 half-pixel 调整
    float srcX = (tidx + 0.5f) * (float)pInWidth / (float)pOutWidth - 0.5f;
    float srcY = (tidy + 0.5f) * (float)pInHeight / (float)pOutHeight - 0.5f;

    srcX = fmaxf(0.0f, fminf(srcX, (float)(pInWidth - 1)));
    srcY = fmaxf(0.0f, fminf(srcY, (float)(pInHeight - 1)));

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

    // 正确的像素索引对应权重：
    // (x0,y0) -> w00
    // (x1,y0) -> w10
    // (x0,y1) -> w01
    // (x1,y1) -> w11
    int in_y00 = y0 * pInYStride + x0;
    int in_y10 = y0 * pInYStride + x1;
    int in_y01 = y1 * pInYStride + x0;
    int in_y11 = y1 * pInYStride + x1;

    int out_y = tidy * pOutYStride + tidx;

    // 读取并加权（使用寄存器缓存）
    float y00 = (float)pInYData[in_y00];
    float y10 = (float)pInYData[in_y10];
    float y01 = (float)pInYData[in_y01];
    float y11 = (float)pInYData[in_y11];

    float y_val =
        y00 * w00 +
        y10 * w10 +
        y01 * w01 +
        y11 * w11;

    pOutYData[out_y] = (uint8_t) (fminf(fmaxf(y_val, 0.0f), 255.0f));

    // UV：仅在输出偶数像素的偶数行写一次（每2x2 block 写一个 UV）
    if ((tidx & 1) == 0 && (tidy & 1) == 0) {
        int uv_x = tidx / 2;
        int uv_y = tidy / 2;
        int inUVWidth = pInWidth / 2;
        int inUVHeight = pInHeight / 2;
        int outUVWidth = pOutWidth / 2;
        int outUVHeight = pOutHeight / 2;

        if (uv_x < outUVWidth && uv_y < outUVHeight) {

            float src_uv_x = (uv_x + 0.5f) * (float)inUVWidth / (float)outUVWidth - 0.5f;
            float src_uv_y = (uv_y + 0.5f) * (float)inUVHeight / (float)outUVHeight - 0.5f;

            src_uv_x = fmaxf(0.0f, fminf(src_uv_x, (float)(inUVWidth - 1)));
            src_uv_y = fmaxf(0.0f, fminf(src_uv_y, (float)(inUVHeight - 1)));

            int ux0 = (int)src_uv_x;
            int uy0 = (int)src_uv_y;
            int ux1 = min(ux0 + 1, inUVWidth - 1);
            int uy1 = min(uy0 + 1, inUVHeight - 1);

            float udx = src_uv_x - ux0;
            float udy = src_uv_y - uy0;

            float uw00 = (1.0f - udx) * (1.0f - udy);
            float uw10 = udx * (1.0f - udy);
            float uw01 = (1.0f - udx) * udy;
            float uw11 = udx * udy;

            int in_uv00 = uy0 * pInUVStride + ux0 * 2;
            int in_uv10 = uy0 * pInUVStride + ux1 * 2;
            int in_uv01 = uy1 * pInUVStride + ux0 * 2;
            int in_uv11 = uy1 * pInUVStride + ux1 * 2;

            int out_uv = uv_y * pOutUVStride + uv_x * 2;

            float u00 = (float)pInUVData[in_uv00 + 0];
            float v00 = (float)pInUVData[in_uv00 + 1];
            float u10 = (float)pInUVData[in_uv10 + 0];
            float v10 = (float)pInUVData[in_uv10 + 1];
            float u01 = (float)pInUVData[in_uv01 + 0];
            float v01 = (float)pInUVData[in_uv01 + 1];
            float u11 = (float)pInUVData[in_uv11 + 0];
            float v11 = (float)pInUVData[in_uv11 + 1];

            float u_val =
                u00 * uw00 + u10 * uw10 + u01 * uw01 + u11 * uw11;
            float v_val =
                v00 * uw00 + v10 * uw10 + v01 * uw01 + v11 * uw11;

            pOutUVData[out_uv + 0] = (uint8_t) (fminf(fmaxf(u_val, 0.0f), 255.0f));
            pOutUVData[out_uv + 1] = (uint8_t) (fminf(fmaxf(v_val, 0.0f), 255.0f));
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