#pragma once

extern "C" {
    #include <libavcodec/avcodec.h>
    #include <libavformat/avformat.h>
    #include <libavutil/avutil.h>
    #include <libavutil/hwcontext.h>
}

#include "Channel.h"
#include "Consumer.h"

class DecoderConsumer : public Consumer {
public:
    DecoderConsumer(const std::string& codec_name);
    virtual ~DecoderConsumer();
    void setAVCodecParameters(AVCodecParameters* codecpar, AVRational time_base);
    void setChannel(PacketChannel* channel);
    FrameChannel* getChannel2Resize();
    FrameChannel* getChannel2Stitch();
    void start();
    void stop();
    virtual void run();
private:
    AVCodecContext* codec_ctx;
    const AVCodec* codec;
    FrameChannel* m_channel2resize;
    FrameChannel* m_channel2stitch;
    PacketChannel* m_channelFromAVFramePro{nullptr};
};
