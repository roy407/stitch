#pragma once
extern "C" {
    #include <libavcodec/avcodec.h>
    #include <libavutil/avutil.h>
    #include <libavutil/hwcontext.h>
    #include <libavformat/avformat.h>
}

#include <iostream>
#include <stdexcept>
#include <atomic>
#include <thread>
#include <memory>
#include "Consumer.h"
#include "Channel.h"

class JetsonDecoderConsumer : public Consumer {
public:
    JetsonDecoderConsumer();
    virtual ~JetsonDecoderConsumer();
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