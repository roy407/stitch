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

RtspConsumer::RtspConsumer(safe_queue<Packet>& packet, AVCodecParameters** codecpar, AVRational* time_base, const std::string& push_stream_url) : packet_input(packet) {
    m_name += "rtsp";
    this->codecpar = codecpar;
    this->time_base = time_base;
    this->output_url = push_stream_url;
    if (!*codecpar || (*codecpar)->codec_type != AVMEDIA_TYPE_VIDEO) {
        LOG_ERROR("Invalid codec parameters");
        return;
    }
    avformat_alloc_output_context2(&out_ctx, nullptr, "rtsp", output_url.c_str());
    AVStream* out_stream = avformat_new_stream(out_ctx, nullptr);
    avcodec_parameters_copy(out_stream->codecpar, *codecpar);
    out_stream->time_base = *time_base;
    av_opt_set(out_ctx->priv_data, "rtsp_transport", "tcp", 0);
    av_opt_set(out_ctx->priv_data, "muxdelay", "0.1", 0);
    int ret1 = avformat_write_header(out_ctx, NULL);
}
void RtspConsumer::start() {
    TaskManager::start();
}
void RtspConsumer::stop() {
    TaskManager::stop();
}

void RtspConsumer::run() {
    while(running) {
        Packet pkt;
        if(!packet_input.wait_and_pop(pkt)) break;
        int ret = av_interleaved_write_frame(out_ctx, pkt.m_data);
        av_packet_free(&pkt.m_data);
    }
    packet_input.clear();
}

RtspConsumer::~RtspConsumer() {
    LOG_DEBUG("{} exit!",__func__);
}