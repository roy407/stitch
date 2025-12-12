#include "StitchConsumer.h"
#include "StitchImpl.h"
#include "resize.cuh"

StitchConsumer::StitchConsumer(StitchOps* ops, int single_width, int height, int output_width) {
    m_name += "stitch";
    this->ops = ops;
    this->single_width = single_width;
    this->height = height;
    this->output_width = output_width;
    m_status.width = output_width;
    m_status.height = height;
    m_channel2show = new FrameChannel;

    hw_frames_ctx = av_hwframe_ctx_alloc(cuda_handle_init::GetGPUDeviceHandle());
    AVHWFramesContext* frames_ctx = (AVHWFramesContext*)hw_frames_ctx->data;
    frames_ctx->format = AV_PIX_FMT_CUDA;
    frames_ctx->sw_format = AV_PIX_FMT_NV12;   // CUDA 支持的底层格式
    frames_ctx->width = 8192;
    frames_ctx->height = 2160;
    frames_ctx->initial_pool_size = 20;

    if (av_hwframe_ctx_init(hw_frames_ctx) < 0) {
        throw std::runtime_error("Failed to initialize CUDA hwframe context");
    }
}

void StitchConsumer::setChannels(std::vector<FrameChannel*> channels) {
    m_channelsFromDecoder = channels;
}

FrameChannel *StitchConsumer::getChannel2Show() {
    return m_channel2show;
}

StitchConsumer::~StitchConsumer() {
    delete m_channel2show;
}

void StitchConsumer::start() {
    TaskManager::start();
}

void StitchConsumer::stop() {
    for(auto& i: m_channelsFromDecoder) {
        i->stop();
    }
    TaskManager::stop();
}

void StitchConsumer::run() { 
    Frame out_image;
    Frame resizeout;
    AVFrame** inputs = new AVFrame*[10];
    while (running) {
        int frame_size = 0;
        for (auto& channel : m_channelsFromDecoder) {
            Frame tmp;
            if(!channel->recv(tmp)) goto cleanup;
            inputs[frame_size] = tmp.m_data;
            out_image.m_costTimes.image_frame_cnt[tmp.cam_id] = tmp.m_costTimes.image_frame_cnt[tmp.cam_id];
            out_image.m_costTimes.when_get_packet[tmp.cam_id] = tmp.m_costTimes.when_get_packet[tmp.cam_id];
            out_image.m_costTimes.when_get_decoded_frame[tmp.cam_id] = tmp.m_costTimes.when_get_decoded_frame[tmp.cam_id];
            frame_size ++;
        }
        out_image.m_data = ops->stitch(ops->obj, inputs);

        resizeout.m_data = av_frame_alloc();
        resizeout.m_data->format = AV_PIX_FMT_CUDA;
        resizeout.m_data->width = 8192;
        resizeout.m_data->height = 2160;
        resizeout.m_data->hw_frames_ctx = av_buffer_ref(hw_frames_ctx);
        if (av_hwframe_get_buffer(hw_frames_ctx, resizeout.m_data, 0) < 0) {
            throw std::runtime_error("Failed to allocate GPU AVFrame buffer");
        }
        uint8_t* output_y = resizeout.m_data->data[0];
        uint8_t* output_uv = resizeout.m_data->data[1];
        int output_linesize_y = resizeout.m_data->linesize[0];
        int output_linesize_uv = resizeout.m_data->linesize[1];

        // out_image.m_data->pts = inputs[0]->pts;
        resizeout.m_data->pts = inputs[0]->pts;
        resizeout.m_costTimes.when_get_stitched_frame = get_now_time();

        ReSize(out_image.m_data->data[0], out_image.m_data->data[1],
        out_image.m_data->width, out_image.m_data->height,
        out_image.m_data->linesize[0], out_image.m_data->linesize[1],
        output_y, output_uv,
        8192, 2160,
        output_linesize_y, output_linesize_uv,
        0);
        cudaStreamSynchronize(0);

        av_frame_free(&out_image.m_data);
        m_channel2show->send(resizeout);
        m_status.frame_cnt ++;
        m_status.timestamp = get_now_time();
        for (int i = 0; i < m_channelsFromDecoder.size(); ++i) {
            if (inputs[i]) {
                av_frame_free(&inputs[i]);
            }
        }
    }
cleanup:
    for(auto& channel : m_channelsFromDecoder) {
        channel->clear();
    }
    delete[] inputs;
}
