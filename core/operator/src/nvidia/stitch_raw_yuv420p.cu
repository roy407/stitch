#include <cuda_runtime.h>
#include <stdint.h>

__global__ void stitch_y_uv_with_linesize_kernel_yuv420p(
    uint8_t* const* inputs_y, uint8_t* const* inputs_u, uint8_t* const* inputs_v,
    int* input_linesize_y, int* input_linesize_u, int* input_linesize_v,
    uint8_t* output_y, uint8_t* output_uv,
    int output_linesize_y, int output_linesize_uv,
    int cam_num, int single_width, int width, int height) {
    
    int cam_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if(cam_idx >= cam_num || row >= height) return;

    // 处理Y分量
    uint8_t* input_line = inputs_y[cam_idx] + row * input_linesize_y[cam_idx];
    uint8_t* output_line = output_y + row * output_linesize_y + cam_idx * single_width;
    for(int i=0; i<single_width; i++) {
        output_line[i] = input_line[i];
    }

    // 处理UV分量(每两行Y对应一行UV)
    if(row < height / 2) {
        // YUV420P中UV平面的宽度也是Y平面的一半
        int uv_width = single_width / 2;
        
        // 获取U和V平面的输入行
        uint8_t* u_input = inputs_u[cam_idx] + row * input_linesize_u[cam_idx];
        uint8_t* v_input = inputs_v[cam_idx] + row * input_linesize_v[cam_idx];
        
        // 计算输出位置（NV12格式：UV交替存储）
        uint8_t* uv_output = output_uv + row * output_linesize_uv + cam_idx * single_width;
        
        // 将分开的U和V平面交错存储为NV12格式
        for(int i = 0; i < uv_width; i++) {
            uv_output[2 * i] = u_input[i];     // U分量
            uv_output[2 * i + 1] = v_input[i]; // V分量
        }
    }
}

extern "C"
void launch_stitch_kernel_raw_yuv420p(
                            uint8_t** inputs_y, uint8_t** inputs_u, uint8_t** inputs_v,
                            int* input_linesize_y, int* input_linesize_u, int* input_linesize_v,
                            uint8_t* output_y, uint8_t* output_uv,
                            int output_linesize_y, int output_linesize_uv,
                            int cam_num, int single_width, int width, int height,
                            cudaStream_t stream) {
    int max_threads_per_block;
    cudaDeviceGetAttribute(&max_threads_per_block, cudaDevAttrMaxThreadsPerBlock, 0);
    dim3 block(16, max_threads_per_block / 16); 
    dim3 grid(
        (cam_num + block.x - 1) / block.x,
        (height + block.y - 1) / block.y
    );
    stitch_y_uv_with_linesize_kernel_yuv420p<<<grid, block, 0, stream>>>(
        inputs_y, inputs_u, inputs_v,
        input_linesize_y, input_linesize_u, input_linesize_v,
        output_y, output_uv,
        output_linesize_y, output_linesize_uv,
        cam_num, single_width, width, height
    );
}