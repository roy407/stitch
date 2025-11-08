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

__global__ void stitch_withH(
    uint8_t* const* inputs_y, uint8_t* const* inputs_uv,
    int* input_linesize_y, int* input_linesize_uv,
    float* h_matrix, uint8_t* output_y, uint8_t* output_uv,
    int output_linesize_y, int output_linesize_uv,
    int cam_num, int single_width, int width, int height)
{
    int cam_idx = blockIdx.z;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;


    if (x >= single_width || y >= height || cam_idx >= cam_num) return;

    float* H = &h_matrix[cam_idx * 9];

    float out_x, out_y;
    applyHomography(H, x, y, &out_x, &out_y);


    if (out_x >= 0 && out_x < width && out_y >= 0 && out_y < height) {

        int out_x_int = __float2int_rn(out_x);
        int out_y_int = __float2int_rn(out_y);
        output_y[out_y_int * output_linesize_y + out_x_int] = inputs_y[cam_idx][y * input_linesize_y[cam_idx] + x];

        if (x % 2 == 0 && y % 2 == 0) {
            float uv_x = x / 2;  
            float uv_y = y / 2;
            int out_uv_x = out_x_int / 2;
            int out_uv_y = out_y_int / 2;

            if (out_uv_x >= 0 && out_uv_x < width/2 && out_uv_y >= 0 && out_uv_y < height/2) {

            int src_uv_x = __float2int_rn(uv_x);  
            int src_uv_y = __float2int_rn(uv_y);
    
            int uv_pitch = input_linesize_uv[cam_idx];
            uint8_t u_val = inputs_uv[cam_idx][src_uv_y * uv_pitch + src_uv_x * 2];      
            uint8_t v_val = inputs_uv[cam_idx][src_uv_y * uv_pitch + src_uv_x * 2 + 1];  
    
            int uv_index = out_uv_y * output_linesize_uv + out_uv_x * 2;
            output_uv[uv_index] = u_val;
            output_uv[uv_index + 1] = v_val;
            }
        }
    }
}
