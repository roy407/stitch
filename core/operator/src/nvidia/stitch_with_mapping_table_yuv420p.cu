#include <cuda_runtime.h>
#include <stdint.h>
#include <iostream>
using namespace std;

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
    if (cam_id >= cam_num || map_x >= single_width || map_y >= height) {
        output_y[y * output_linesize_y + x] = 0;
        return;
    }

    // Y channel
    uint8_t* input_y = inputs_y[cam_id];
    int in_pitch_y = input_linesize_y[cam_id];
    uint8_t val_y = input_y[map_y * in_pitch_y + map_x];
    output_y[y * output_linesize_y + x] = val_y;
}

__global__ void stitch_kernel_UV_with_mapping_table(
    uint8_t** inputs_u, int* input_linesize_u, uint8_t** inputs_v, int* input_linesize_v,
    uint8_t* output_uv, int output_linesize_uv,
    int cam_num, int single_width, int width, int height, const cudaTextureObject_t mapping_table)
{
    int x = (blockIdx.x * blockDim.x + threadIdx.x) << 1;
    int y = (blockIdx.y * blockDim.y + threadIdx.y) << 1;

    if (x >= width || y >= height)
        return;

    ushort4 entry = tex2D<ushort4>(mapping_table, x, y);
    uint16_t cam_id = entry.x;
    uint16_t map_x  = entry.y;
    uint16_t map_y  = entry.z;

    if (cam_id >= cam_num || map_x >= single_width || map_y >= height) {
        int uv_out_x = (x & ~1);
        int uv_offset = (y >> 1) * output_linesize_uv + uv_out_x;
        output_uv[uv_offset]     = 128;
        output_uv[uv_offset + 1] = 128;
        return;
    }

    // UV channel
    uint8_t* input_u = inputs_u[cam_id];
    int in_pitch_u = input_linesize_u[cam_id];
    uint8_t* input_v = inputs_v[cam_id];
    int in_pitch_v = input_linesize_v[cam_id];

    // 计算偏移
    int u_offset_in  = (map_y >> 1) * in_pitch_u + (map_x >> 1);
    int v_offset_in  = (map_y >> 1) * in_pitch_v + (map_x >> 1);
    int uv_offset_out = (y >> 1) * output_linesize_uv + (x & ~1);

    uint8_t u = 128, v = 128;
    u = input_u[u_offset_in];
    v = input_v[v_offset_in];
    output_uv[uv_offset_out]     = u;
    output_uv[uv_offset_out + 1] = v;
}

extern "C"
void launch_stitch_kernel_with_mapping_table_yuv420p(
    uint8_t** inputs_y, uint8_t** inputs_u, uint8_t** inputs_v,
    int* input_linesize_y, int* input_linesize_u, int* input_linesize_v,
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
        inputs_u, input_linesize_u, inputs_v, input_linesize_v,
        output_uv, output_linesize_uv,
        cam_num, single_width, width, height, mapping_table);
}