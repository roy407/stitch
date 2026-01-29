#include "SingleViewConsumer.h"
#include "cuda_handle_init.h"
#include "resize.cuh"

// 构造函数中不能调用其他构造函数，否则只是创建了临时对象，应该像现在这样，以委托构造的方式
SingleViewConsumer::SingleViewConsumer(int width, int height, float scale_factor) : SingleViewConsumer(width,
    height,
    int(width * scale_factor),
    int(height * scale_factor)) {
}

SingleViewConsumer::SingleViewConsumer(int width, int height, AVRational rational) : SingleViewConsumer(width,
    height,
    int(width * rational.num / rational.den),
    int(height * rational.num / rational.den)) {

}

SingleViewConsumer::SingleViewConsumer(int width, int height, int output_width, int output_height) {
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
    frames_ctx->initial_pool_size = 5;

    if (av_hwframe_ctx_init(hw_frames_ctx) < 0) {
        throw std::runtime_error("Failed to initialize CUDA hwframe context");
    }
    cudaStreamCreate(&stream);

    m_channel2show = new FrameChannel;
}

void SingleViewConsumer::start() {
    TaskManager::start();
}

void SingleViewConsumer::stop() {
    m_channelFromDecoder->stop();
    TaskManager::stop();
}

void SingleViewConsumer::run() {
    while (running) {
        Frame tmp;
        Frame out_image;
        if(!m_channelFromDecoder->recv(tmp)) goto cleanup;

        // 当前的resize会出现抖动问题，暂时注释掉！！！
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

        ReSize(input_y, input_uv,
        width, height,
        input_linesize_y, input_linesize_uv,
        output_y, output_uv,
        output_width, output_height,
        output_linesize_y, output_linesize_uv,
        stream);
        cudaStreamSynchronize(stream);
        out_image.m_data->pts = tmp.m_data->pts;
        
        m_channel2show->send(out_image);
        av_frame_free(&tmp.m_data);
    }
cleanup:
    m_channelFromDecoder->clear();
}

SingleViewConsumer::~SingleViewConsumer() {
    cudaStreamDestroy(stream);
    delete m_channel2show;
}

void SingleViewConsumer::setChannel(FrameChannel *channel) {
    m_channelFromDecoder = channel;
}

FrameChannel *SingleViewConsumer::getChannel2Show() const {
    return m_channel2show;
}
