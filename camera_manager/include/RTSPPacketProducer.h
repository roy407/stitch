#pragma once

#include <memory>

extern "C" {
    #include <libavcodec/avcodec.h>
    #include <libavformat/avformat.h>
}

#include "config.h"

#include "Channel.h"
#include "LogConsumer.h"
#include "PacketProducer.h"

class RTSPPacketProducer : public PacketProducer {
protected:
    AVFormatContext* fmt_ctx{nullptr};
    AVDictionary* options{nullptr};
    std::string cam_path;
    int video_stream{-1};
public:
    RTSPPacketProducer(CameraConfig camera_config);
    RTSPPacketProducer(int cam_id, std::string name, std::string input_url, int width, int height);
    virtual ~RTSPPacketProducer();
    virtual void start();
    virtual void stop();
    virtual void run();
};