#pragma once

#include "config.h"

#include "PacketProducer.h"

class MP4PacketProducer : public PacketProducer {
protected:
    AVFormatContext* fmt_ctx{nullptr};
    AVDictionary* options{nullptr};
    std::string cam_path;
    int video_stream{-1};
public:
    MP4PacketProducer(CameraConfig camera_config);
    virtual ~MP4PacketProducer();
    virtual void start();
    virtual void stop();
    virtual void run();
};