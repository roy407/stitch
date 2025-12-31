# pragma once
#include "Consumer.h"
#include <vector>
extern "C" {
    #include "libavformat/avformat.h"
    #include "libavcodec/avcodec.h"
    #include "libavutil/pixfmt.h" 
    #include "libavutil/pixdesc.h" 
    #include "libavutil/opt.h"
    #include "libavutil/log.h"
    #include "libavcodec/bsf.h"
}

#include "safe_queue.hpp"
#include "tools.hpp"
#include "config.h"
#include "EncoderConsumer.h"
#include "safe_queue.hpp"
#include "LogConsumer.h"
#include "Channel.h"

class StitchOps; // 提前声明



class StitchConsumer : public Consumer {
    std::vector<FrameChannel*> m_channelsFromDecoder;
    FrameChannel* m_channel2show;
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
    virtual ~StitchConsumer();
    virtual void start();
    virtual void stop();
    virtual void run();
};