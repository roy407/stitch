#include "RtspConsumer.h"
#include <iostream>
#include <unistd.h>
#include <sys/wait.h>
#include <fcntl.h>
// #include "safe_queue.hpp"
#include "list_queue.hpp"
extern "C" {
    #include <libavformat/avformat.h>
    #include <libavcodec/avcodec.h>
    #include <libavutil/opt.h>
    #include <libavutil/log.h>
}
#include "log.hpp"

RtspConsumer::RtspConsumer(const std::string &push_stream_url) {
    m_name += "rtsp";
    this->output_url = push_stream_url;
}
void RtspConsumer::setChannel(PacketChannel *m_channel) {
    m_channelFromAVFramePro = m_channel;
}
void RtspConsumer::setParamters(AVCodecParameters *codecpar, AVRational time_base) {
    this->codecpar = codecpar;
    this->time_base = time_base;
    if (!codecpar || (codecpar)->codec_type != AVMEDIA_TYPE_VIDEO) {
        LOG_ERROR("Invalid codec parameters");
        return;
    }
    avformat_alloc_output_context2(&out_ctx, nullptr, "rtsp", output_url.c_str());
    AVStream* out_stream = avformat_new_stream(out_ctx, nullptr);
    avcodec_parameters_copy(out_stream->codecpar, codecpar);
    out_stream->time_base = time_base;
    av_opt_set(out_ctx->priv_data, "rtsp_transport", "tcp", 0);
    av_opt_set(out_ctx->priv_data, "muxdelay", "0.1", 0);
    int ret = avformat_write_header(out_ctx, NULL);
}
void RtspConsumer::start() {
    TaskManager::start();
}
void RtspConsumer::stop() {
    m_channelFromAVFramePro->stop();
    TaskManager::stop();
}

void RtspConsumer::run() {
    while(running) {
        Packet pkt;
        if(!m_channelFromAVFramePro->recv(pkt)) break;
        int ret = av_interleaved_write_frame(out_ctx, pkt.m_data);
        av_packet_free(&pkt.m_data);
    }
    m_channelFromAVFramePro->clear();
}

RtspConsumer::~RtspConsumer() {
    LOG_DEBUG("{} exit!",__func__);
}