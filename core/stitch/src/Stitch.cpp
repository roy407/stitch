#include "Stitch.h"
#include "stitch.cuh"
#include "scale.cuh"
#include <iostream>

#include <libavutil/frame.h>
#include <libavutil/hwcontext.h>
#include <libavutil/pixfmt.h>

#include "cuda_handle_init.h"
#include "config.h"

Stitch::Stitch(int width,int height,int cam_num): single_width(width),height(height),cam_num(cam_num) {
    
    output_width = single_width * cam_num;
    int* crop = new int[cam_num * 4];
    memset(crop,0,cam_num * 4);
    const std::vector<CameraConfig> cams = config::GetInstance().GetCameraConfig();
    for(int i=0;i<cams.size();i++) {
        if(cams[i].stitch.enable == true) {
            if(cams[i].stitch.mode == "crop") {
                std::vector<float> __crop = cams[i].crop;
                crop[i*4] = __crop[0] * width;
                crop[i*4+1] = __crop[1] * height;
                crop[i*4+2] = __crop[2] * width;
                crop[i*4+3] = __crop[3] * height;
            }
        }
    }
    
    //TODO：已经计算出了裁剪区域，应该可以直接得到输出图像的大小

    cudaMalloc(&d_inputs_y, sizeof(uint8_t*) * cam_num);
    cudaMalloc(&d_inputs_uv, sizeof(uint8_t*) * cam_num);
    cudaMalloc(&d_crop, cam_num * 4 * sizeof(int));
    cudaMemcpy(d_crop, crop, cam_num * sizeof(int), cudaMemcpyHostToDevice);
    // 创建 HW frame context
    hw_frames_ctx = av_hwframe_ctx_alloc(cuda_handle_init::GetGPUDeviceHandle());
    AVHWFramesContext* frames_ctx = (AVHWFramesContext*)hw_frames_ctx->data;
    frames_ctx->format = AV_PIX_FMT_CUDA;
    frames_ctx->sw_format = AV_PIX_FMT_NV12;   // CUDA 支持的底层格式
    frames_ctx->width = output_width;
    frames_ctx->height = height;
    frames_ctx->initial_pool_size = 20;

    if (av_hwframe_ctx_init(hw_frames_ctx) < 0) {
        throw std::runtime_error("Failed to initialize CUDA hwframe context");
    }
    delete[] crop;
    running.store(true);
}

Stitch::~Stitch() {
    running.store(false);
    cudaFree(d_inputs_y);
    cudaFree(d_inputs_uv);
    cudaFree(d_crop);
}

AVFrame* Stitch::do_stitch(AVFrame** inputs) {

    uint8_t* gpu_inputs_y[cam_num];
    uint8_t* gpu_inputs_uv[cam_num];
    for (int i = 0; i < cam_num; ++i) {
        if (!inputs[i]) {
            return nullptr;
        }
        gpu_inputs_y[i] = inputs[i]->data[0];
        gpu_inputs_uv[i] = inputs[i]->data[1];
    }

    output = av_frame_alloc();
    if (!output) {
        throw std::runtime_error("Failed to allocate output frame");
    }

    output->format = AV_PIX_FMT_CUDA;
    output->width = output_width;
    output->height = height;

    output->hw_frames_ctx = av_buffer_ref(hw_frames_ctx);

    if (av_hwframe_get_buffer(hw_frames_ctx, output, 0) < 0) {
        throw std::runtime_error("Failed to allocate GPU AVFrame buffer");
    }

    cudaMemcpy(d_inputs_y, gpu_inputs_y, sizeof(uint8_t*) * cam_num, cudaMemcpyHostToDevice);
    cudaMemcpy(d_inputs_uv, gpu_inputs_uv, sizeof(uint8_t*) * cam_num, cudaMemcpyHostToDevice);

    uint8_t* output_y = output->data[0];
    uint8_t* output_uv = output->data[1];

    cudaStream_t stream = 0;

    /*如果是不变量，考虑只初始化一次---待修改*/
    int h_input_linesize_y[cam_num];
    int h_input_linesize_uv[cam_num];

    for(int i=0;i<cam_num;i++) {
        h_input_linesize_uv[i] = inputs[i]->linesize[1];
        h_input_linesize_y[i] = inputs[i]->linesize[0];
    }

    int* d_input_linesize_y;
    int* d_input_linesize_uv;
    cudaMalloc(&d_input_linesize_y, cam_num * sizeof(int));
    cudaMalloc(&d_input_linesize_uv, cam_num * sizeof(int));
    cudaMemcpy(d_input_linesize_y, h_input_linesize_y, cam_num * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input_linesize_uv, h_input_linesize_uv, cam_num * sizeof(int), cudaMemcpyHostToDevice);

    launch_scale_1_2_kernel(d_inputs_y, d_inputs_uv, d_input_linesize_y, d_input_linesize_uv,
                        single_width, height, cam_num, stream); 

    launch_stitch_kernel_with_crop(d_inputs_y, d_inputs_uv,
                        d_input_linesize_y, d_input_linesize_uv,
                        output_y, output_uv,
                        output->linesize[0], output->linesize[1],
                        cam_num, single_width, output_width, height,
                        stream,d_crop);
    
    return output;
}
