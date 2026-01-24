#pragma once

#include <string>

extern "C" {
    #include <libavcodec/avcodec.h>
    #include <libavformat/avformat.h>
    #include <libavutil/log.h>
    #include <libavutil/opt.h>
}

#include "Channel.h"
#include "Consumer.h"

class RtspConsumer : public Consumer {
public:
    RtspConsumer(const std::string& push_stream_url);
    void setChannel(PacketChannel* m_channel);
    void setParameters(AVCodecContext* enc_ctx); 
    void setParameters(AVCodecParameters* codecpar, AVRational time_base);
    virtual void start();
    virtual void stop();
    virtual void run();
    virtual ~RtspConsumer();
private:
    std::string m_url;
    PacketChannel* m_input_channel{nullptr};
    AVFormatContext* m_out_ctx{nullptr};
    AVStream* m_out_stream{nullptr};
    AVRational m_in_time_base{1, 25};
};