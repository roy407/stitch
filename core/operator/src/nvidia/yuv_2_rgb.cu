#include "yuv_2_rgb.cuh"
#include <cuda_runtime.h>

__global__ void NV12ToRGBAKernel(uint8_t* d_y, uint8_t* d_uv, uchar4* d_rgba, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int y_idx = y * width + x;
    int uv_idx = (y / 2) * (width / 2 * 2) + (x / 2) * 2;

    float Y = (float)d_y[y_idx];
    float U = (float)d_uv[uv_idx] - 128.0f;
    float V = (float)d_uv[uv_idx + 1] - 128.0f;

    float R = Y + 1.402f * V;
    float G = Y - 0.344136f * U - 0.714136f * V;
    float B = Y + 1.772f * U;

    R = fminf(fmaxf(R, 0.0f), 255.0f);
    G = fminf(fmaxf(G, 0.0f), 255.0f);
    B = fminf(fmaxf(B, 0.0f), 255.0f);

    d_rgba[y_idx] = make_uchar4((unsigned char)R, (unsigned char)G, (unsigned char)B, 255);
}

extern "C" void convertNV12ToRGBA(uint8_t* d_y, uint8_t* d_uv, uchar4* d_rgba, int width, int height)
{
    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);
    NV12ToRGBAKernel<<<grid, block>>>(d_y, d_uv, d_rgba, width, height);
    cudaDeviceSynchronize();
}