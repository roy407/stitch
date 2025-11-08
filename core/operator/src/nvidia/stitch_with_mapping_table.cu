#include <cuda_runtime.h>
#include <stdint.h>
#include <iostream>
using namespace std;

__global__ void stitch_kernel_with_mapping_table(
    uint8_t** inputs_y, uint8_t** inputs_uv,
    int* input_linesize_y, int* input_linesize_uv,
    uint8_t* output_y, uint8_t* output_uv,
    int output_linesize_y, int output_linesize_uv,
    int cam_num, int single_width, int width, int height,
    const uint16_t* mapping_table)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int idx = (x * height + y) * 3;

    uint16_t cam_id = mapping_table[idx + 0];
    uint16_t map_x  = mapping_table[idx + 1];
    uint16_t map_y  = mapping_table[idx + 2];

    if (cam_id >= cam_num) {
        output_y[y * output_linesize_y + x] = 0;
        if ((y % 2 == 0) && (x % 2 == 0)) {
            int uv_out_x = (x & ~1);
            int uv_offset = (y / 2) * output_linesize_uv + uv_out_x;
            output_uv[uv_offset]     = 128;
            output_uv[uv_offset + 1] = 128;
        }
        return;
    }

    // Y channel
    uint8_t* input_y = inputs_y[cam_id];
    int in_pitch_y = input_linesize_y[cam_id];
    uint8_t val_y = input_y[map_y * in_pitch_y + map_x];
    output_y[y * output_linesize_y + x] = val_y;

    // UV channel (NV12)
    // 只在每个 2x2 块的左上角写一次
    if ((y % 2 == 0) && (x % 2 == 0)) {
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
}

extern "C"
void launch_stitch_kernel_with_mapping_table(
    uint8_t** inputs_y, uint8_t** inputs_uv,
    int* input_linesize_y, int* input_linesize_uv,
    uint8_t* output_y, uint8_t* output_uv,
    int output_linesize_y, int output_linesize_uv,
    int cam_num, int single_width, int width, int height, const uint16_t* mapping_table,
    cudaStream_t stream) {
    
    dim3 block(16,16); 
    dim3 grid(
        (width + block.x - 1) / block.x,  // 水平方向块数
        (height + block.y - 1) / block.y  // 垂直方向块数
    );

    stitch_kernel_with_mapping_table<<<grid, block, 0, stream>>>(
        inputs_y, inputs_uv,
        input_linesize_y, input_linesize_uv,
        output_y, output_uv,
        output_linesize_y, output_linesize_uv,
        cam_num, single_width, width, height, mapping_table
    );
}