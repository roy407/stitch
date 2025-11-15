#include <cuda_runtime.h>
#include <stdint.h>

__device__ void applyHomography(float* H, float x, float y, float* out_x, float* out_y) {
    float denominator = H[6]*x + H[7]*y + H[8];
    if (fabsf(denominator) < 1e-6f) {
        *out_x = -1;
        *out_y = -1;
        return;
    }
    *out_x = (H[0]*x + H[1]*y + H[2]) / denominator;
    *out_y = (H[3]*x + H[4]*y + H[5]) / denominator;
}

__global__ void stitch_kernel_with_h_matrix_inv(
    uint8_t* const* inputs_y, uint8_t* const* inputs_uv,
    int* input_linesize_y, int* input_linesize_uv,
    float* h_matrix_inv, uint8_t* output_y, uint8_t* output_uv,
    int output_linesize_y, int output_linesize_uv,
    int cam_num, int single_width, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    uint16_t y = 0;
    uint16_t u = 0;
    uint16_t v = 0;
    uint16_t match_cnt = 0;
    // 单应性变换和采样
    for(uint16_t active_cam = cam_num - 1;active_cam >= 0; active_cam --) {
        float* H_inv = &h_matrix_inv[active_cam * 9];
        float __x, __y;
        applyHomography(H_inv, x - 6339, y, &__x, &__y);
        if(__x != -1) {
            int src_x_int = __float2int_rn(__x);
            int src_y_int = __float2int_rn(__y);
            y += inputs_y[active_cam][src_y_int * input_linesize_y[active_cam] + src_x_int];
            match_cnt += 1;
            if(x % 2 == 0 && y % 2 == 0) {
                float uv_x = __x / 2.0f;
                float uv_y = __y / 2.0f;
                int src_uv_x = __float2int_rn(uv_x);
                int src_uv_y = __float2int_rn(uv_y);
                u += inputs_uv[active_cam][src_uv_y * input_linesize_uv[active_cam] + src_uv_x * 2];
                v += inputs_uv[active_cam][src_uv_y * input_linesize_uv[active_cam] + src_uv_x * 2 + 1];
            }
        }
    }
    if(match_cnt != 0) {
        y /= match_cnt;
        u /= match_cnt;
        v /= match_cnt;
    }
    output_y[y * output_linesize_y + x] = y;

    if(x % 2 == 0 && y % 2 == 0) {
        int out_uv_idx = (y/2) * output_linesize_uv + (x/2) * 2;
        output_uv[out_uv_idx] = u;
        output_uv[out_uv_idx + 1] = v;
    }
}

extern "C"
void launch_stitch_kernel_with_h_matrix_inv(
    uint8_t** inputs_y, uint8_t** inputs_uv,
    int* input_linesize_y, int* input_linesize_uv,float* h_matrix_inv, float** cam_polygons,
    uint8_t* output_y, uint8_t* output_uv,
    int output_linesize_y, int output_linesize_uv,
    int cam_num, int single_width, int width, int height,
    cudaStream_t stream) {
    int max_threads_per_block;
    cudaDeviceGetAttribute(&max_threads_per_block, cudaDevAttrMaxThreadsPerBlock, 0);
    dim3 block(16, max_threads_per_block / 16); 
    dim3 grid(
        (width + block.x - 1) / block.x,  // 水平方向块数
        (height + block.y - 1) / block.y        // 垂直方向块数
    );

    stitch_kernel_with_h_matrix_inv<<<grid, block, 0, stream>>>(
        inputs_y, inputs_uv,
        input_linesize_y, input_linesize_uv,h_matrix_inv, cam_polygons,
        output_y, output_uv,
        output_linesize_y, output_linesize_uv,
        cam_num, single_width, width, height
    );
}