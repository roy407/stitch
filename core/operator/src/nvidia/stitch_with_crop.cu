#include <cuda_runtime.h>
#include <stdint.h>

__global__ void stitch_y_uv_with_linesize_and_crop_kernel(
    uint8_t* const* inputs_y, uint8_t* const* inputs_uv,
    int* input_linesize_y, int* input_linesize_uv,
    uint8_t* output_y, uint8_t* output_uv,
    int output_linesize_y, int output_linesize_uv,
    int cam_num, int single_width, int width, int height,
    int* crop)  
{
    int cam_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if(cam_idx >= cam_num || row >= height) return;

    int sum=0;
    for(int i=0;i<cam_idx;i++){
        sum +=crop[i];
    }

    int cropped_width = single_width - crop[cam_idx];
    int output_offset = cam_idx * single_width - sum;
    

    uint8_t* input_line = inputs_y[cam_idx] + row * input_linesize_y[cam_idx];
    uint8_t* output_line = output_y + row * output_linesize_y + output_offset;

    for(int i = crop[cam_idx]; i < single_width; i++) {
        output_line[i-crop[cam_idx]] = input_line[i];
    }

    if(row < height/2) {
        uint8_t* uv_input = inputs_uv[cam_idx] + row * input_linesize_uv[cam_idx];
        uint8_t* uv_output = output_uv + row * output_linesize_uv + output_offset;
        
        for(int i = crop[cam_idx]; i < single_width; i++) {
            uv_output[i - crop[cam_idx]] = uv_input[i];
        }
    }
}

extern "C"
void launch_stitch_kernel_with_crop(
    uint8_t** inputs_y, uint8_t** inputs_uv,
    int* input_linesize_y, int* input_linesize_uv,
    uint8_t* output_y, uint8_t* output_uv,
    int output_linesize_y, int output_linesize_uv,
    int cam_num, int single_width, int width, int height,
    cudaStream_t stream, int* crop) {
    int max_threads_per_block;
    cudaDeviceGetAttribute(&max_threads_per_block, cudaDevAttrMaxThreadsPerBlock, 0);
    dim3 block(16, max_threads_per_block / 16); 
    dim3 grid((cam_num + block.x - 1) / block.x, 
              (height + block.y - 1) / block.y);

    stitch_y_uv_with_linesize_and_crop_kernel<<<grid, block, 0, stream>>>(
        inputs_y, inputs_uv,
        input_linesize_y, input_linesize_uv,
        output_y, output_uv,
        output_linesize_y, output_linesize_uv,
        cam_num, single_width, width, height,
        crop); 
}