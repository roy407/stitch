#include <cuda_runtime.h>
#include <stdint.h>

__device__ void applyHomography(float* H, float x, float y, float* out_x, float* out_y) {
    float denominator = H[6]*x + H[7]*y + H[8];
    if (fabsf(denominator) < 1e-6f) {
        *out_x = -255;
        *out_y = -255;
        return;
    }
    *out_x = (H[0]*x + H[1]*y + H[2]) / denominator;
    *out_y = (H[3]*x + H[4]*y + H[5]) / denominator;
}

__global__ void stitch_kernel_Y_with_h_matrix_inv(
    uint8_t** inputs_y, int* input_linesize_y,
    uint8_t* output_y, int output_linesize_y,
    int cam_num, int single_width, int width, int height,
    float* h_matrix_inv)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;
    uint16_t Y = 0;
    uint16_t match_cnt = 0;
    // 单应性变换和采样
    for(int active_cam = cam_num - 1;active_cam >= 0; active_cam --) {
        float* H_inv = &h_matrix_inv[active_cam * 9];
        float __x, __y;
        applyHomography(H_inv, x - 3840 * 3, y, &__x, &__y);
        int src_x_int = __float2int_rn(__x);
        int src_y_int = __float2int_rn(__y);
        if (src_x_int >= 0 && src_x_int < single_width && src_y_int >= 0 && src_y_int < height) {
            Y += inputs_y[active_cam][src_y_int * input_linesize_y[active_cam] + src_x_int];
            match_cnt += 1;
        }
    }
    if(match_cnt != 0) {
        Y /= match_cnt;
    }
    output_y[y * output_linesize_y + x] = Y;
}

__global__ void stitch_kernel_UV_with_h_matrix_inv(
    uint8_t** inputs_uv, int* input_linesize_uv,
    uint8_t* output_uv, int output_linesize_uv,
    int cam_num, int single_width, int width, int height,
    float* h_matrix_inv)
{
    int x = (blockIdx.x * blockDim.x + threadIdx.x) << 1;
    int y = (blockIdx.y * blockDim.y + threadIdx.y) << 1;

    if (x >= width || y >= height) return;
    uint16_t U = 0;
    uint16_t V = 0;
    uint16_t match_cnt = 0;
    // 单应性变换和采样
    for(int active_cam = cam_num - 1;active_cam >= 0; active_cam --) {
        float* H_inv = &h_matrix_inv[active_cam * 9];
        float __x, __y;
        applyHomography(H_inv, x - 3840 * 3, y, &__x, &__y);
        float uv_x = __x / 2.0f;
        float uv_y = __y / 2.0f;
        int src_uv_x = __float2int_rn(uv_x);
        int src_uv_y = __float2int_rn(uv_y);
        if (src_uv_x >= 0 && src_uv_x < single_width / 2 && src_uv_y >= 0 && src_uv_y < height / 2) {
            U += inputs_uv[active_cam][src_uv_y * input_linesize_uv[active_cam] + (src_uv_x << 1)];
            V += inputs_uv[active_cam][src_uv_y * input_linesize_uv[active_cam] + (src_uv_x << 1) + 1];
            match_cnt += 1;
        }
    }
    if(match_cnt != 0) {
        U /= match_cnt;
        V /= match_cnt;
    }
    int out_uv_idx = (y >> 1) * output_linesize_uv + x;
    output_uv[out_uv_idx] = U;
    output_uv[out_uv_idx + 1] = V;
}

extern "C"
void launch_stitch_kernel_with_h_matrix_inv(uint8_t** inputs_y, uint8_t** inputs_uv, 
    int* input_linesize_y, int* input_linesize_uv, 
    uint8_t* output_y, uint8_t* output_uv,
    int output_linesize_y, int output_linesize_uv,
    int cam_num, int single_width, int width, int height, float* h_matrix_inv, cudaStream_t stream1, cudaStream_t stream2) {
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

    stitch_kernel_Y_with_h_matrix_inv<<<gridY, block, 0, stream1>>>(
        inputs_y, input_linesize_y,
        output_y, output_linesize_y,
        cam_num, single_width, width, height, h_matrix_inv);

    stitch_kernel_UV_with_h_matrix_inv<<<gridUV, block, 0, stream2>>>(
        inputs_uv, input_linesize_uv,
        output_uv, output_linesize_uv,
        cam_num, single_width, width, height, h_matrix_inv);
}