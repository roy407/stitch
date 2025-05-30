#include "Stitch.h"
#include "stitch.cuh"

#include <iostream>

#include <libavutil/frame.h>
#include <libavutil/hwcontext.h>
#include <libavutil/pixfmt.h>

#include "cuda_handle_init.h"

Stitch::Stitch(int cam_num,int single_width,int height): cam_num(cam_num),single_width(single_width),height(height) {
    
    const int output_width = single_width * cam_num;
    size = cam_num;

    // 创建 HW frame context
    hw_frames_ctx = av_hwframe_ctx_alloc(cuda_handle_init::GetGPUDeviceHandle());
    AVHWFramesContext* frames_ctx = (AVHWFramesContext*)hw_frames_ctx->data;
    frames_ctx->format = AV_PIX_FMT_CUDA;
    frames_ctx->sw_format = AV_PIX_FMT_NV12;   // CUDA 支持的底层格式
    frames_ctx->width = output_width;
    frames_ctx->height = height;
    frames_ctx->initial_pool_size = 1;

    if (av_hwframe_ctx_init(hw_frames_ctx) < 0) {
        throw std::runtime_error("Failed to initialize CUDA hwframe context");
    }

    running.store(true);
}

Stitch::~Stitch() {
    running.store(false);
}

AVFrame* Stitch::do_stitch(AVFrame** inputs) {

    const int output_width = single_width * cam_num;

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

    uint8_t **d_inputs_y, **d_inputs_uv;
    cudaMalloc(&d_inputs_y, sizeof(uint8_t*) * cam_num);
    cudaMalloc(&d_inputs_uv, sizeof(uint8_t*) * cam_num);

    cudaMemcpy(d_inputs_y, gpu_inputs_y, sizeof(uint8_t*) * cam_num, cudaMemcpyHostToDevice);
    cudaMemcpy(d_inputs_uv, gpu_inputs_uv, sizeof(uint8_t*) * cam_num, cudaMemcpyHostToDevice);

    uint8_t* output_y = output->data[0];
    uint8_t* output_uv = output->data[1];

    cudaStream_t stream = 0;

    launch_stitch_y_uv_kernel(d_inputs_y, d_inputs_uv, output_y, output_uv,
                              cam_num, single_width, output_width, height, stream);

    cudaFree(d_inputs_y);
    cudaFree(d_inputs_uv);


    return output;
}