#include "scale.cuh"
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <iostream>

// NV12 scale 0.5 (separate in/out)
__global__ void scale_1_2_y_uv_kernel(
    const uint8_t* __restrict__ input_y,
    const uint8_t* __restrict__ input_uv,
    uint8_t* __restrict__ output_y,
    uint8_t* __restrict__ output_uv,
    int src_w,
    int src_h,
    int src_ls_y,
    int src_ls_uv,
    int dst_ls_y,
    int dst_ls_uv)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;  // output x
    int y = blockIdx.y * blockDim.y + threadIdx.y;  // output y

    int dst_w = src_w / 2;
    int dst_h = src_h / 2;

    // ---------- Y ----------
    if (x < dst_w && y < dst_h) {
        int src_x = x * 2;
        int src_y = y * 2;

        const uint8_t* r1 = input_y + src_y * src_ls_y + src_x;
        const uint8_t* r2 = input_y + (src_y + 1) * src_ls_y + src_x;

        uint8_t v = (r1[0] + r1[1] + r2[0] + r2[1]) >> 2;

        output_y[y * dst_ls_y + x] = v;
    }

    // ---------- UV ----------
    if (x < dst_w / 2 && y < dst_h / 2) {
        int src_x = x * 4;
        int src_y = y * 2;

        const uint8_t* uv1 = input_uv + src_y * src_ls_uv + src_x;
        const uint8_t* uv2 = input_uv + (src_y + 1) * src_ls_uv + src_x;

        uint8_t u = (uv1[0] + uv1[2] + uv2[0] + uv2[2]) >> 2;
        uint8_t v = (uv1[1] + uv1[3] + uv2[1] + uv2[3]) >> 2;

        uint8_t* out_uv = output_uv + y * dst_ls_uv + x * 2;
        out_uv[0] = u;
        out_uv[1] = v;
    }
}


extern "C" void launch_scale_1_2_kernel(
    const uint8_t* input_y, const uint8_t* input_uv,
    int input_linesize_y, int input_linesize_uv,
    uint8_t* output_y, uint8_t* output_uv,
    int output_linesize_y, int output_linesize_uv,
    int src_w, int src_h, cudaStream_t stream)
{
    int dst_w = src_w / 2;
    int dst_h = src_h / 2;

    int max_threads_per_block;
    cudaDeviceGetAttribute(&max_threads_per_block, cudaDevAttrMaxThreadsPerBlock, 0);

    dim3 block(16, max_threads_per_block / 16); 
    dim3 grid((dst_w + block.x - 1) / block.x,
              (dst_h + block.y - 1) / block.y);

    scale_1_2_y_uv_kernel<<<grid, block, 0, stream>>>(
        input_y, input_uv,
        output_y, output_uv,
        src_w, src_h,
        input_linesize_y, input_linesize_uv,
        output_linesize_y, output_linesize_uv
    );
}

// NV12 scale 0.16667 (separate in/out)
__global__ void scale_1_6_y_uv_kernel(
    const uint8_t* __restrict__ input_y,
    const uint8_t* __restrict__ input_uv,
    uint8_t* __restrict__ output_y,
    uint8_t* __restrict__ output_uv,
    int src_w,
    int src_h,
    int src_ls_y,
    int src_ls_uv,
    int dst_ls_y,
    int dst_ls_uv)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;  // output x
    int y = blockIdx.y * blockDim.y + threadIdx.y;  // output y

    int dst_w = src_w / 6;
    int dst_h = src_h / 6;

    // ---------- Y ----------
    if (x < dst_w && y < dst_h) {
        int src_x = x * 6;
        int src_y = y * 6;

        const uint8_t* r1 = input_y + src_y * src_ls_y + src_x;
        const uint8_t* r2 = input_y + (src_y + 1) * src_ls_y + src_x;
        const uint8_t* r3 = input_y + (src_y + 2) * src_ls_y + src_x;
        const uint8_t* r4 = input_y + (src_y + 3) * src_ls_y + src_x;
        const uint8_t* r5 = input_y + (src_y + 4) * src_ls_y + src_x;
        const uint8_t* r6 = input_y + (src_y + 5) * src_ls_y + src_x;
        uint64_t t = r1[0] + r1[1] + r1[2] + r1[3] + r1[4] + r1[5];
        t += r2[0] + r2[1] + r2[2] + r2[3] + r2[4] + r2[5];
        t += r3[0] + r3[1] + r3[2] + r3[3] + r3[4] + r3[5];
        t += r4[0] + r4[1] + r4[2] + r4[3] + r4[4] + r4[5];
        t += r5[0] + r5[1] + r5[2] + r5[3] + r5[4] + r5[5];
        t += r6[0] + r6[1] + r6[2] + r6[3] + r6[4] + r6[5];
        uint8_t v = t / 36;

        output_y[y * dst_ls_y + x] = v;
    }

    // ---------- UV ----------
    if (x < dst_w / 2 && y < dst_h / 2) {
        int src_x = x * 12;
        int src_y = y * 6;

        const uint8_t* uv1 = input_uv + src_y * src_ls_uv + src_x;
        const uint8_t* uv2 = input_uv + (src_y + 1) * src_ls_uv + src_x;
        const uint8_t* uv3 = input_uv + (src_y + 2) * src_ls_uv + src_x;
        const uint8_t* uv4 = input_uv + (src_y + 3) * src_ls_uv + src_x;
        const uint8_t* uv5 = input_uv + (src_y + 4) * src_ls_uv + src_x;
        const uint8_t* uv6 = input_uv + (src_y + 5) * src_ls_uv + src_x;
        uint64_t u_t = uv1[0] + uv1[2] + uv1[4];
        u_t += uv2[0] + uv2[2] + uv2[4];
        u_t += uv3[0] + uv3[2] + uv3[4];
        u_t += uv4[0] + uv4[2] + uv4[4];
        u_t += uv5[0] + uv5[2] + uv5[4];
        u_t += uv6[0] + uv6[2] + uv6[4];
        uint64_t v_t = uv1[1] + uv1[3] + uv1[5];
        v_t += uv2[1] + uv2[3] + uv2[5];
        v_t += uv3[1] + uv3[3] + uv3[5];
        v_t += uv4[1] + uv4[3] + uv4[5];
        v_t += uv5[1] + uv5[3] + uv5[5];
        v_t += uv6[1] + uv6[3] + uv6[5];
        uint8_t u = u_t / 18;
        uint8_t v = v_t / 18;

        uint8_t* out_uv = output_uv + y * dst_ls_uv + x * 2;
        out_uv[0] = u;
        out_uv[1] = v;
    }
}


extern "C" void launch_scale_1_6_kernel(
    const uint8_t* input_y, const uint8_t* input_uv,
    int input_linesize_y, int input_linesize_uv,
    uint8_t* output_y, uint8_t* output_uv,
    int output_linesize_y, int output_linesize_uv,
    int src_w, int src_h, cudaStream_t stream)
{
    int dst_w = src_w / 6;
    int dst_h = src_h / 6;

    int max_threads_per_block;
    cudaDeviceGetAttribute(&max_threads_per_block, cudaDevAttrMaxThreadsPerBlock, 0);

    dim3 block(16, max_threads_per_block / 16); 
    dim3 grid((dst_w + block.x - 1) / block.x,
              (dst_h + block.y - 1) / block.y);

    scale_1_6_y_uv_kernel<<<grid, block, 0, stream>>>(
        input_y, input_uv,
        output_y, output_uv,
        src_w, src_h,
        input_linesize_y, input_linesize_uv,
        output_linesize_y, output_linesize_uv
    );
}

