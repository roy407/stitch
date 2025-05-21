#include "Stitch.h"
#include "stitch.cuh"

#include <iostream>

Stitch::Stitch() {
    
    const int cam_num = 5;
    const int single_width = 3840;
    const int height = 2160;
    const int output_width = single_width * cam_num;

    AVBufferRef* hw_device_ctx = nullptr;
    av_hwdevice_ctx_create(&hw_device_ctx, AV_HWDEVICE_TYPE_CUDA, nullptr, nullptr, 0);

    // 创建 HW frame context
    hw_frames_ctx = av_hwframe_ctx_alloc(hw_device_ctx);
    AVHWFramesContext* frames_ctx = (AVHWFramesContext*)hw_frames_ctx->data;
    frames_ctx->format = AV_PIX_FMT_CUDA;
    frames_ctx->sw_format = AV_PIX_FMT_NV12;   // CUDA 支持的底层格式
    frames_ctx->width = output_width;
    frames_ctx->height = height;
    frames_ctx->initial_pool_size = 1;

    if (av_hwframe_ctx_init(hw_frames_ctx) < 0) {
        throw std::runtime_error("Failed to initialize CUDA hwframe context");
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

    running.store(true);
}

Stitch::~Stitch() {
    av_frame_free(&output);
    running.store(false);
}

AVFrame* Stitch::do_stitch(AVFrame** inputs) {
    const int cam_num = size;
    const int single_width = 3840;
    const int height = 2160;
    const int output_width = single_width * cam_num;

    uint8_t* gpu_inputs[cam_num];
    for (int i = 0; i < cam_num; ++i) {
        if(!inputs[i]) {
            std::cout<< "input is nullptr" <<std::endl;
            return nullptr;
        }
        gpu_inputs[i] = inputs[i]->data[0];
    }

    uint8_t* output_gpu_ptr = output->data[0];
    cudaStream_t stream = 0;
    launch_stitch_kernel(gpu_inputs, output_gpu_ptr, output_width, height, stream);
    return output;
}