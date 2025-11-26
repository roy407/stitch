#include <cuda_runtime.h>
#include <stdint.h>
#include <iostream>
using namespace std;



#define KERNEL_RADIUS 2           
#define KERNEL_SIZE (2*KERNEL_RADIUS + 1)

__constant__ float gauss_kernel[KERNEL_SIZE*KERNEL_SIZE] = {
    // 1, 4, 6, 4, 1,
    // 4,16,24,16, 4,
    // 6,24,36,24, 6,
    // 4,16,24,16, 4,
    // 1, 4, 6, 4, 1
    12, 24, 36, 24, 12,
    24,8,9,8, 24,
    36,9,1,9, 36,
    24,8,9,8, 24,
    12, 24, 36, 24, 12
};

__device__ int get_region_index(int x)
{
    const int region_x0[7] = {2995, 5466, 7539, 10309, 12776, 15349, 17452};
    const int region_x1[7] = {3493, 5834, 7975, 10479, 13018, 15671, 17981};
    for(int r=0;r<7;r++){
        if(x >= region_x0[r] && x < region_x1[r]) return r;
    }
    return -1;
}

__global__ void stitch_kernel_Y_with_mapping_table(
    uint8_t** inputs_y, int* input_linesize_y,
    uint8_t* output_y, int output_linesize_y,
    int cam_num, int single_width, int width, int height,
    const cudaTextureObject_t mapping_table)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    
    ushort4 entry = tex2D<ushort4>(mapping_table, x, y);
    uint16_t cam_id = entry.x;
    uint16_t map_x  = entry.y;
    uint16_t map_y  = entry.z;
    if (cam_id >= cam_num) {
        output_y[y * output_linesize_y + x] = 255;
        return;
    }

    // Y channel
    uint8_t* input_y = inputs_y[cam_id];
    int in_pitch_y = input_linesize_y[cam_id];
    uint8_t val_y = input_y[map_y * in_pitch_y + map_x];

    output_y[y * output_linesize_y + x] = val_y;

}

__global__ void stitch_kernel_UV_with_mapping_table(
    uint8_t** inputs_uv, int* input_linesize_uv,
    uint8_t* output_uv, int output_linesize_uv,
    int cam_num, int single_width, int width, int height, const cudaTextureObject_t mapping_table)
{
    int x = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    int y = (blockIdx.y * blockDim.y + threadIdx.y) * 2;

    if (x >= width || y >= height)
        return;

    ushort4 entry = tex2D<ushort4>(mapping_table, x, y);
    uint16_t cam_id = entry.x;
    uint16_t map_x  = entry.y;
    uint16_t map_y  = entry.z;

    if (cam_id >= cam_num) {
        int uv_out_x = (x & ~1);
        int uv_offset = (y / 2) * output_linesize_uv + uv_out_x;
        output_uv[uv_offset]     = 128;
        output_uv[uv_offset + 1] = 128;
        return;
    }

    // UV channel (NV12)
    uint8_t* input_uv = inputs_uv[cam_id];
    int in_pitch_uv = input_linesize_uv[cam_id];

    // 对 map_x 对齐到偶数（取偶索引）
    int map_x_uv = (map_x & ~1);
    int uv_in_row = map_y / 2;
    int uv_out_row = y / 2;

    // 计算偏移
    int uv_offset_in  = uv_in_row * in_pitch_uv + map_x_uv;
    int uv_offset_out = uv_out_row * output_linesize_uv + (x & ~1);

    uint8_t u = 128, v = 128;
    u = input_uv[uv_offset_in];
    v = input_uv[uv_offset_in + 1];
    output_uv[uv_offset_out]     = u;
    output_uv[uv_offset_out + 1] = v;
}

extern "C"
void launch_stitch_kernel_with_mapping_table(
    uint8_t** inputs_y, uint8_t** inputs_uv,
    int* input_linesize_y, int* input_linesize_uv,
    uint8_t* output_y, uint8_t* output_uv,
    int output_linesize_y, int output_linesize_uv,
    int cam_num, int single_width, int width, int height, const cudaTextureObject_t mapping_table,
    cudaStream_t stream1, cudaStream_t stream2) {
    
    int max_threads_per_block;
    cudaDeviceGetAttribute(&max_threads_per_block, cudaDevAttrMaxThreadsPerBlock, 0);
    dim3 block(16, max_threads_per_block / 16);
    dim3 gridY(
        (width + block.x - 1) / block.x,  // 水平方向块数
        (height + block.y - 1) / block.y  // 垂直方向块数
    );
    dim3 gridUV(
        (width / 2 + block.x - 1) / block.x,  // 水平方向块数
        (height / 2 + block.y - 1) / block.y  // 垂直方向块数
    );

    stitch_kernel_Y_with_mapping_table<<<gridY, block, 0, stream1>>>(
        inputs_y, input_linesize_y,
        output_y, output_linesize_y,
        cam_num, single_width, width, height, mapping_table);

    stitch_kernel_UV_with_mapping_table<<<gridUV, block, 0, stream2>>>(
        inputs_uv, input_linesize_uv,
        output_uv, output_linesize_uv,
        cam_num, single_width, width, height, mapping_table);
}