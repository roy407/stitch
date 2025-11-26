#pragma once
#include "config.h"
#include "AVFrameProducer.h"

class AVFrameProducer_debug : public AVFrameProducer {
public:
    AVFrameProducer_debug(CameraConfig camera_config);
    virtual void run();
};