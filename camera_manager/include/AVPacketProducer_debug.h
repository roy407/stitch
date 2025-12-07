#pragma once
#include "config.h"
#include "AVPacketProducer.h"

class AVPacketProducer_debug : public AVPacketProducer {
public:
    AVPacketProducer_debug(CameraConfig camera_config);
    virtual void run();
};