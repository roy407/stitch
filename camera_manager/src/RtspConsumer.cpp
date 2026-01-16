#include "RtspConsumer.h"
#include <iostream>
#include <unistd.h>
#include <sys/wait.h>
#include <fcntl.h>
#include "safe_queue.hpp"
extern "C" {
    #include <libavformat/avformat.h>
    #include <libavcodec/avcodec.h>
    #include <libavutil/opt.h>
    #include <libavutil/log.h>
}
#include "log.hpp"

RtspConsumer::RtspConsumer(const std::string &push_stream_url) {
    m_name += "rtsp";
    m_url = push_stream_url; 
}
void RtspConsumer::setChannel(PacketChannel *m_channel) {
    m_input_channel = m_channel;
}
void RtspConsumer::setParameters(AVCodecContext* enc_ctx) {
    if (!enc_ctx) {
        LOG_ERROR("Invalid codec context");
        return;
    }
    int ret = avformat_alloc_output_context2(&m_out_ctx, nullptr, "rtsp", m_url.c_str());
    if (ret < 0) {
        LOG_ERROR("Failed to allocate output context");
        return;
    }
    m_out_stream = avformat_new_stream(m_out_ctx, nullptr);
    if (!m_out_stream) {
        LOG_ERROR("Failed to create new stream");
        return;
    }
    if (avcodec_parameters_from_context(m_out_stream->codecpar, enc_ctx) < 0) {
        LOG_ERROR("Failed to copy codec parameters");
        return;
    }
    m_out_stream->time_base = enc_ctx->time_base;
    m_in_time_base = enc_ctx->time_base;
    av_opt_set(m_out_ctx->priv_data, "rtsp_transport", "tcp", 0);
    av_opt_set(m_out_ctx->priv_data, "muxdelay", "0.1", 0);
    if (!(m_out_ctx->oformat->flags & AVFMT_NOFILE)) {
        if (avio_open(&m_out_ctx->pb, m_url.c_str(), AVIO_FLAG_WRITE) < 0) {
            LOG_ERROR("Could not open output URL: {}", m_url);
            return;
        }
    }
    if (avformat_write_header(m_out_ctx, nullptr) < 0) {
        LOG_ERROR("Error occurred when writing header to output url");
    }
}

void RtspConsumer::setParameters(AVCodecParameters* codecpar, AVRational time_base) {
     if (!codecpar) {
        LOG_ERROR("Invalid codec parameters");
        return;
    }
    int ret = avformat_alloc_output_context2(&m_out_ctx, nullptr, "rtsp", m_url.c_str());
    if (ret < 0) {
        LOG_ERROR("Failed to allocate output context");
        return;
    }
    m_out_stream = avformat_new_stream(m_out_ctx, nullptr);
    if (!m_out_stream) {
        LOG_ERROR("Failed to create new stream");
        return;
    }
    
    if (avcodec_parameters_copy(m_out_stream->codecpar, codecpar) < 0) {
        LOG_ERROR("Failed to copy codec parameters");
        return;
    }
    m_out_stream->time_base = time_base;
    m_in_time_base = time_base;
    av_opt_set(m_out_ctx->priv_data, "rtsp_transport", "tcp", 0);
    av_opt_set(m_out_ctx->priv_data, "muxdelay", "0.1", 0);
    if (!(m_out_ctx->oformat->flags & AVFMT_NOFILE)) {
        if (avio_open(&m_out_ctx->pb, m_url.c_str(), AVIO_FLAG_WRITE) < 0) {
            LOG_ERROR("Could not open output URL: {}", m_url);
            return;
        }
    }
    if (avformat_write_header(m_out_ctx, nullptr) < 0) {
        LOG_ERROR("Error occurred when writing header to output url");
    }
}
void RtspConsumer::start() {
    TaskManager::start();
}
void RtspConsumer::stop() {
    m_input_channel->stop();
    TaskManager::stop();
}
void RtspConsumer::run() {
    Packet pkt;
    while(running) {
        if(!m_input_channel->recv(pkt)) {
            LOG_INFO("RtspConsumer input channel closed");
            break; 
        }
        if (pkt.m_data && m_out_ctx) {
            av_packet_rescale_ts(pkt.m_data, 
                m_in_time_base, 
                m_out_stream->time_base);
            av_interleaved_write_frame(m_out_ctx, pkt.m_data);
            av_packet_free(&pkt.m_data);
        }
    }
    m_input_channel->clear();    
    if (m_out_ctx) {
        av_write_trailer(m_out_ctx);
        if (!(m_out_ctx->oformat->flags & AVFMT_NOFILE)) avio_closep(&m_out_ctx->pb);
        avformat_free_context(m_out_ctx);
    }
}
RtspConsumer::~RtspConsumer() {
    stop();
    LOG_DEBUG("{} exit!",__func__);
}