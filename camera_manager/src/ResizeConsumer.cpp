#include "ResizeConsumer.h"
#include "cuda_handle_init.h"
#include "scale.cuh"

// 构造函数中不能调用其他构造函数，否则只是创建了临时对象，应该像现在这样，以委托构造的方式
ResizeConsumer::ResizeConsumer(int width, int height, float scale_factor) : ResizeConsumer(width,
    height,
    int(width * scale_factor),
    int(height * scale_factor)) {
}

ResizeConsumer::ResizeConsumer(int width, int height, AVRational rational) : ResizeConsumer(width,
    height,
    int(width * rational.num / rational.den),
    int(height * rational.num / rational.den)) {

}

ResizeConsumer::ResizeConsumer(int width, int height, int output_width, int output_height) {
    this->width = width;
    this->height = height;
    this->output_width = output_width;
    this->output_height = output_height;
    m_name += "resize";

    hw_frames_ctx = av_hwframe_ctx_alloc(cuda_handle_init::GetGPUDeviceHandle());
    AVHWFramesContext* frames_ctx = (AVHWFramesContext*)hw_frames_ctx->data;
    frames_ctx->format = AV_PIX_FMT_CUDA;
    frames_ctx->sw_format = AV_PIX_FMT_NV12;   // CUDA 支持的底层格式
    frames_ctx->width = output_width;
    frames_ctx->height = output_height;
    frames_ctx->initial_pool_size = 20;

    if (av_hwframe_ctx_init(hw_frames_ctx) < 0) {
        throw std::runtime_error("Failed to initialize CUDA hwframe context");
    }
}

void ResizeConsumer::SetInputFrame(safe_queue<Frame> *InputFrame) {
    this->InputFrame = InputFrame;
}

void ResizeConsumer::start() {
    TaskManager::start();
}

void ResizeConsumer::stop() {
    if(InputFrame) InputFrame->stop();
    TaskManager::stop();
}

void ResizeConsumer::run() {
    while (running) {
        Frame tmp;
        Frame out_image;
        if(!InputFrame->wait_and_pop(tmp)) goto cleanup;
        const uint8_t* input_y = tmp.m_data->data[0];
        const uint8_t* input_uv = tmp.m_data->data[1];
        int input_linesize_y = tmp.m_data->linesize[0];
        int input_linesize_uv = tmp.m_data->linesize[1];
        out_image.m_data = av_frame_alloc();
        out_image.m_data->format = AV_PIX_FMT_CUDA;
        out_image.m_data->width = output_width;
        out_image.m_data->height = output_height;
        out_image.m_data->hw_frames_ctx = av_buffer_ref(hw_frames_ctx);
        if (av_hwframe_get_buffer(hw_frames_ctx, out_image.m_data, 0) < 0) {
            throw std::runtime_error("Failed to allocate GPU AVFrame buffer");
        }
        uint8_t* output_y = out_image.m_data->data[0];
        uint8_t* output_uv = out_image.m_data->data[1];
        int output_linesize_y = out_image.m_data->linesize[0];
        int output_linesize_uv = out_image.m_data->linesize[1];

        cudaStream_t stream = 0;
        launch_scale_1_2_kernel(input_y, input_uv,
        input_linesize_y, input_linesize_uv,
        output_y, output_uv,
        output_linesize_y, output_linesize_uv,
        width, height, stream);
        cudaStreamSynchronize(stream);
        out_image.m_data->pts = tmp.m_data->pts;
        OutputFrame.push(out_image);
        av_frame_free(&tmp.m_data);
    }
cleanup:
    InputFrame->clear();
}

ResizeConsumer::~ResizeConsumer() {

}

safe_queue<Frame> &ResizeConsumer::get_resize_frame() {
    return OutputFrame;
}
