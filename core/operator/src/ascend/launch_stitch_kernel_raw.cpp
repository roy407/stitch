
// #include "stitch.h"
#include <stdint.h>
#include <stdio.h>
#include <iostream>
#
#include <kernel_operator.h>

extern "C" __global__ __aicore__ void stitch_y_uv_with_linesize_kernel(
    __gm__ uint8_t** inputs_y, __gm__ uint8_t** inputs_uv,
    __gm__ int* input_linesize_y, __gm__ int* input_linesize_uv,
    __gm__ uint8_t* output_y, __gm__ uint8_t* output_uv,
    int output_linesize_y, int output_linesize_uv,
    int cam_num, int single_width, int width, int height) {
    
    // 昇腾任务划分：每个task处理一个摄像头的两行Y和一行UV
    int32_t task_id = get_block_idx();
    int32_t total_tasks = cam_num * (height / 2);  // 总任务数 = 摄像头数 * (高度/2)
    
    if (task_id >= total_tasks) return;

    // 计算摄像头索引和行组索引
    int32_t cam_idx = task_id / (height / 2);
    int32_t row_group = task_id % (height / 2);  // 每组包含两行Y和一行UV

    // 获取当前摄像头的输入指针
    __gm__ uint8_t* input_y_ptr = inputs_y[cam_idx];
    __gm__ uint8_t* input_uv_ptr = inputs_uv[cam_idx];
    
    // 计算输入行偏移
    int32_t input_y_offset0 = row_group * 2 * input_linesize_y[cam_idx];
    int32_t input_y_offset1 = (row_group * 2 + 1) * input_linesize_y[cam_idx];
    int32_t input_uv_offset = row_group * input_linesize_uv[cam_idx];
    
    // 计算输出位置
    int32_t output_y_offset0 = (row_group * 2) * output_linesize_y + cam_idx * single_width;
    int32_t output_y_offset1 = (row_group * 2 + 1) * output_linesize_y + cam_idx * single_width;
    int32_t output_uv_offset = row_group * output_linesize_uv + cam_idx * single_width * 2;  // UV每个像素2字节

    // 拷贝Y分量第一行
    for(int i=0;i<single_width;i++) {
        *(output_y + output_y_offset0 + i) = *(input_y_ptr + input_y_offset0 + i);
    }
    
    // 拷贝Y分量第二行
    for(int i=0;i<single_width;i++) {
        *(output_y + output_y_offset1 + i) = *(input_y_ptr + input_y_offset1 + i);
    }
    
    // 拷贝UV分量（NV12格式，每像素2字节）
    for(int i=0;i<single_width * 2;i++) {
        *(output_uv + output_uv_offset + i) = *(input_uv_ptr + input_uv_offset + i);
    }
}

extern "C"
void launch_stitch_kernel_raw(
    uint8_t** inputs_y, uint8_t** inputs_uv,
    int* input_linesize_y, int* input_linesize_uv,
    uint8_t* output_y, uint8_t* output_uv,
    int output_linesize_y, int output_linesize_uv,
    int cam_num, int single_width, int width, int height,
    void* stream) {
    int total_tasks = cam_num * (height / 2);
    stitch_y_uv_with_linesize_kernel<<<total_tasks, nullptr, stream>>>(
        inputs_y, inputs_uv,
        input_linesize_y, input_linesize_uv,
        output_y, output_uv,
        output_linesize_y, output_linesize_uv,
        cam_num, single_width, width, height
    );
}
