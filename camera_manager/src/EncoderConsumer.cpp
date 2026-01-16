#include "EncoderConsumer.h"
#include<iostream>
#include "cuda_handle_init.h"
#include "log.hpp"

extern "C" {
    #include <libavutil/opt.h>
    #include <libavutil/imgutils.h>
}


EncoderConsumer::EncoderConsumer(const std::string& codec_name, int width, int height, int fps)
    : m_width(width), m_height(height), m_fps(fps) {
    m_name = codec_name + "_encoder";
    codec = avcodec_find_encoder_by_name(codec_name.c_str());
    CHECK_NULL(codec);

    codec_ctx = avcodec_alloc_context3(codec);

    CHECK_NULL(codec_ctx);
    codec_ctx->width = width;
    codec_ctx->height = height;
    codec_ctx->time_base = AVRational{1, fps};
    codec_ctx->framerate = AVRational{fps, 1};
    codec_ctx->gop_size = 12;
    codec_ctx->max_b_frames = 0;
    codec_ctx->pix_fmt = AV_PIX_FMT_CUDA; 

    if (codec->id == AV_CODEC_ID_H264 || codec->id == AV_CODEC_ID_HEVC) {
        av_opt_set(codec_ctx->priv_data, "preset", "p1", 0);
        av_opt_set(codec_ctx->priv_data, "gpu", "0", 0);
        av_opt_set(codec_ctx->priv_data, "tune", "ull", 0);
    }

    codec_ctx->hw_device_ctx = av_buffer_ref(cuda_handle_init::GetGPUDeviceHandle());

    AVBufferRef* frames_ref = av_hwframe_ctx_alloc(cuda_handle_init::GetGPUDeviceHandle());
    CHECK_NULL(frames_ref); 
    
    AVHWFramesContext* frames_ctx = (AVHWFramesContext*)(frames_ref->data);
    frames_ctx->format = AV_PIX_FMT_CUDA;
    frames_ctx->sw_format = AV_PIX_FMT_NV12;
    frames_ctx->width = width;
    frames_ctx->height = height;
    frames_ctx->initial_pool_size = 20;

    int ret = av_hwframe_ctx_init(frames_ref);
    CHECK_FFMPEG_RETURN(ret); 
    
    codec_ctx->hw_frames_ctx = av_buffer_ref(frames_ref);
    av_buffer_unref(&frames_ref);
}

EncoderConsumer::~EncoderConsumer() {
    stop();
    if (codec_ctx) {
        if (codec_ctx->hw_device_ctx) {
            av_buffer_unref(&codec_ctx->hw_device_ctx);
        }
        avcodec_free_context(&codec_ctx);
    }
}

void EncoderConsumer::setInputChannel(FrameChannel* channel) {
    CHECK_NULL(channel);
    m_input_channel = channel;
}

void EncoderConsumer::setOutputChannel(PacketChannel* channel) {
    CHECK_NULL(channel);
    m_output_channel = channel;
}

void EncoderConsumer::start() {
    if(m_input_channel == nullptr || m_output_channel == nullptr) {
        LOG_ERROR("EncoderConsumer input or output channel has not inited");
        return;
    }
    int ret = avcodec_open2(codec_ctx, codec, nullptr);
    if (ret < 0) {
        char errbuf[128];
        av_strerror(ret, errbuf, sizeof(errbuf));
        LOG_ERROR("Failed to open codec: {} (code: {})", errbuf, ret);
        running = false;
        return;
    }
    TaskManager::start();
}

void EncoderConsumer::stop() {
    if (m_input_channel) {
        m_input_channel->stop();
    }
    TaskManager::stop();
}

void EncoderConsumer::run(){
    if (!codec_ctx || !avcodec_is_open(codec_ctx)) {
        LOG_ERROR("Codec context not open, exiting run loop");
        return;
    }
    CHECK_NULL_RETURN(m_input_channel);
    CHECK_NULL_RETURN(m_output_channel);
    Frame frame;
    AVPacket *av_pkt = av_packet_alloc();
    CHECK_NULL_RETURN(av_pkt);
    Packet packet;
    LOG_INFO("Encoder thread started");
    int64_t pts_counter = 0;
    while (running) {
        if(!m_input_channel->recv(frame)) {
            LOG_INFO("Input channel received exit signal");
            break;
        }
        if(frame.m_data == nullptr) continue;
        frame.m_data->pts = pts_counter++;
        int ret = avcodec_send_frame(codec_ctx, frame.m_data);
        if (frame.m_data) {
             av_frame_free(&frame.m_data);
        }

        if (ret < 0) {
            CHECK_FFMPEG_RETURN(ret);
            continue;
        }

        while (ret >= 0) {
            ret = avcodec_receive_packet(codec_ctx, av_pkt);

            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                break;
            } else if (ret < 0) {   
                CHECK_FFMPEG_RETURN(ret);
                break;
            }
            packet.m_data = av_packet_alloc();
            CHECK_NULL_RETURN(packet.m_data);
            if(packet.m_data) {
                av_packet_ref(packet.m_data, av_pkt);
                packet.m_costTimes = frame.m_costTimes;
                packet.cam_id = frame.cam_id;
                packet.m_timestamp = frame.m_timestamp;
                m_output_channel->send(packet);
            }
            av_packet_unref(av_pkt);
        }
        
    }
    av_packet_free(&av_pkt);
    LOG_INFO("Encoder thread finished.");

}