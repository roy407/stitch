#pragma once

#include <string>
#include <vector>

extern "C" {
    #include <libavcodec/avcodec.h>
    #include <libavcodec/bsf.h>
    #include <libavformat/avformat.h>
    #include <libavutil/log.h>
    #include <libavutil/opt.h>
    #include <libavutil/pixdesc.h>
    #include <libavutil/pixfmt.h>
}

#include "config.h"
#include "tools.hpp"

#include "safe_queue.hpp"

#include "Channel.h"
#include "Consumer.h"

#include "EncoderConsumer.h"
#include "LogConsumer.h"

class StitchOps;

class StitchConsumer : public Consumer {
    std::vector<FrameChannel*> m_channelsFromDecoder;
    FrameChannel* m_channel2show = nullptr;
    FrameChannel* m_channel2rtsp = nullptr;
    
    int single_width{0};
    int output_width{0};
    int height{0};
    StitchStatus m_status{};
    std::string url;
    AVFormatContext* out_ctx;
    AVStream* out_stream;
    AVCodecParameters* codecpar;
    StitchOps* ops;
    friend class LogConsumer;
public:
    StitchConsumer(StitchOps* ops, int single_width, int height, int output_width);
    void setChannels(std::vector<FrameChannel*> channels);
    
    FrameChannel* getChannel2Show();
    FrameChannel* getChannel2Rtsp();

    virtual ~StitchConsumer();
    virtual void start();
    virtual void stop();
    virtual void run();
};