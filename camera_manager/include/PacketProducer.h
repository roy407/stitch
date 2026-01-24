#pragma once

#include <memory>

extern "C" {
    #include <libavcodec/avcodec.h>
    #include <libavformat/avformat.h>
}

#include "config.h"

#include "Channel.h"
#include "LogConsumer.h"
#include "Producer.h"

class PacketProducer : public Producer {
protected:
    PacketChannel* m_channel2rtsp{nullptr};
    PacketChannel* m_channel2decoder{nullptr};
    int cam_id{-1};
    AVCodecParameters* codecpar{nullptr};
    AVRational time_base;
    CamStatus m_status{};
    friend class LogConsumer;
public:
    PacketProducer();
    virtual ~PacketProducer();
    virtual void start();
    virtual void stop();
    int getWidth() const;
    int getHeight() const;
    AVRational getTimeBase() const;
    AVCodecParameters* getAVCodecParameters() const;
    PacketChannel* getChannel2Rtsp() const;
    PacketChannel* getChannel2Decoder() const;
};