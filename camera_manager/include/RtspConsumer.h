#pragma once

#include "Consumer.h"
extern "C" {
    #include <libavformat/avformat.h>
    #include <libavcodec/avcodec.h>
    #include <libavutil/opt.h>
    #include <libavutil/log.h>
}

#include "safe_queue.hpp"

class RtspConsumer : public Consumer {
public:
    RtspConsumer(safe_queue<Packet>& packet, AVCodecParameters** codecpar, AVRational* time_base, const std::string& push_stream_url);
    virtual void start();
    virtual void stop();
    virtual void run();
    virtual ~RtspConsumer();
private:
    AVFormatContext* out_ctx{nullptr};
    AVCodecParameters** codecpar{nullptr};
    AVRational* time_base{nullptr};
    safe_queue<Packet>& packet_input;
    std::string output_url;
};