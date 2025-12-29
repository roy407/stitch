#include <cuda_runtime.h>
#include <stdint.h>
//static 限制作用域在本文件内, 避免与其他文件同名函数冲突
static __device__ bool is_point_in_quadrilateral(float x, float y,
    float x1, float y1, float x2, float y2,
    float x3, float y3, float x4, float y4)
{
    // 向量叉积法判断点是否在凸四边形内
    auto cross = [](float ax, float ay, float bx, float by) {
        return ax * by - ay * bx;
    };

    float d1 = cross(x - x1, y - y1, x2 - x1, y2 - y1);
    float d2 = cross(x - x2, y - y2, x3 - x2, y3 - y2);
    float d3 = cross(x - x3, y - y3, x4 - x3, y4 - y3);
    float d4 = cross(x - x4, y - y4, x1 - x4, y1 - y4);

    return (d1 * d3 >= 0) && (d2 * d4 >= 0);
}

static __device__ void applyHomography(float* H, float x, float y, float* out_x, float* out_y) {
    float denominator = H[6]*x + H[7]*y + H[8];
    if (fabsf(denominator) < 1e-6f) {
        *out_x = -1;
        *out_y = -1;
        return;
    }
    *out_x = (H[0]*x + H[1]*y + H[2]) / denominator;
    *out_y = (H[3]*x + H[4]*y + H[5]) / denominator;
}

__global__ void stitch_kernel_Y_with_h_matrix_inv(
    uint8_t** inputs_y, int* input_linesize_y,
    uint8_t* output_y, int output_linesize_y,
    int cam_num, int single_width, int width, int height,
    float* h_matrix_inv, float** cam_polygons)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int active_cam = -1;
    for (int cam = cam_num-1; cam >= 0; --cam) {
        float* quad = cam_polygons[cam];
        if (is_point_in_quadrilateral(x - 6339, y, 
            quad[0], quad[1], quad[2], quad[3], 
            quad[4], quad[5], quad[6], quad[7])) {
            active_cam = cam;
            break;
        }
    }
    if (active_cam == -1) return;

    // --- 单应性变换和采样 ---
    float* H_inv = &h_matrix_inv[active_cam * 9];
    float src_x, src_y;
    applyHomography(H_inv, x - 6339, y, &src_x, &src_y);


    // 处理 Y 通道
    if (src_x >= 0 && src_x < single_width && src_y >= 0 && src_y < height) {
        int src_x_int = __float2int_rn(src_x);
        int src_y_int = __float2int_rn(src_y);
        output_y[y * output_linesize_y + x] = 
            inputs_y[active_cam][src_y_int * input_linesize_y[active_cam] + src_x_int];
    }
}

__global__ void stitch_kernel_UV_with_h_matrix_inv(
    uint8_t** inputs_uv, int* input_linesize_uv,
    uint8_t* output_uv, int output_linesize_uv,
    int cam_num, int single_width, int width, int height,
    float* h_matrix_inv, float** cam_polygons)
{
    int x = (blockIdx.x * blockDim.x + threadIdx.x) << 1;
    int y = (blockIdx.y * blockDim.y + threadIdx.y) << 1;

    if (x >= width || y >= height) return;

    int active_cam = -1;
    for (int cam = cam_num-1; cam >= 0; --cam) {
        float* quad = cam_polygons[cam];
        if (is_point_in_quadrilateral(x - 6339, y, 
            quad[0], quad[1], quad[2], quad[3], 
            quad[4], quad[5], quad[6], quad[7])) {
            active_cam = cam;
            break;
        }
    }
    if (active_cam == -1) return;

    // --- 单应性变换和采样 ---
    float* H_inv = &h_matrix_inv[active_cam * 9];
    float src_x, src_y;
    applyHomography(H_inv, x - 6339, y, &src_x, &src_y);
    float uv_x = src_x / 2.0f;
    float uv_y = src_y / 2.0f;
    
    if (uv_x >= 0 && uv_x < single_width/2 && uv_y >= 0 && uv_y < height/2) {
        int src_uv_x = __float2int_rn(uv_x);
        int src_uv_y = __float2int_rn(uv_y);
        int uv_pitch = input_linesize_uv[active_cam];
        
        int out_uv_idx = (y/2) * output_linesize_uv + (x/2) * 2;
        output_uv[out_uv_idx] = inputs_uv[active_cam][src_uv_y * uv_pitch + src_uv_x * 2];
        output_uv[out_uv_idx + 1] = inputs_uv[active_cam][src_uv_y * uv_pitch + src_uv_x * 2 + 1];
    }
}

extern "C"
void launch_stitch_kernel_with_h_matrix_inv_v1_1(
    uint8_t** inputs_y, uint8_t** inputs_uv,
    int* input_linesize_y, int* input_linesize_uv,
    uint8_t* output_y, uint8_t* output_uv,
    int output_linesize_y, int output_linesize_uv,
    int cam_num, int single_width, int width, int height, float* h_matrix_inv, float** cam_polygons,
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

    stitch_kernel_Y_with_h_matrix_inv<<<gridY, block, 0, stream1>>>(
        inputs_y, input_linesize_y,
        output_y, output_linesize_y,
        cam_num, single_width, width, height, h_matrix_inv, cam_polygons);

    stitch_kernel_UV_with_h_matrix_inv<<<gridUV, block, 0, stream2>>>(
        inputs_uv, input_linesize_uv,
        output_uv, output_linesize_uv,
        cam_num, single_width, width, height, h_matrix_inv, cam_polygons);
}