#pragma once

#include "Producer.h"
#include "Consumer.h"
#include "StitchConsumer.h"
#include "LogConsumer.h"
#include "image_decoder.h"
#include <memory>
extern "C" {
    #include "libavformat/avformat.h"
    #include "libavcodec/avcodec.h"
    #include "libavutil/pixfmt.h" 
    #include "libavutil/pixdesc.h" 
    #include "libavutil/opt.h"
    #include "libavutil/log.h"
    #include "libavcodec/bsf.h"
}
#include "config.h"
#include "ResizeConsumer.h"
#include "AVFrameProducer.h"

class AVFrameProducer_debug : public AVFrameProducer {
    bool inited = false;
public:
    // 目前只有第一个构造函数可以被使用
    AVFrameProducer_debug(CameraConfig camera_config);
    virtual void run();
};