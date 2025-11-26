#pragma once

#include "Consumer.h"
extern "C" {
    #include <libavformat/avformat.h>
    #include <libavcodec/avcodec.h>
    #include <libavutil/opt.h>
    #include <libavutil/log.h>
}

#include "Channel.h"

class RtspConsumer : public Consumer {
public:
    RtspConsumer(const std::string& push_stream_url);
    void setChannel(PacketChannel* m_channel);
    void setParamters(AVCodecParameters* codecpar, AVRational time_base);
    virtual void start();
    virtual void stop();
    virtual void run();
    virtual ~RtspConsumer();
private:
    AVFormatContext* out_ctx{nullptr};
    AVCodecParameters* codecpar{nullptr};
    AVRational time_base;
    PacketChannel* m_channelFromAVFramePro;
    std::string output_url;
};