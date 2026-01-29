#pragma once

#include <linux/videodev2.h>

#include "config.h"

#include "PacketProducer.h"

class USBPacketProducer : public PacketProducer {
protected:
    std::string cam_path;
    AVFormatContext *fmt_ctx{nullptr};
    AVDictionary *options{nullptr};
    int video_stream{0};
    const AVInputFormat* iformat{nullptr};
public:
    USBPacketProducer(CameraConfig camera_config);
    virtual ~USBPacketProducer();
    virtual void start();
    virtual void stop();
    virtual void run();
};