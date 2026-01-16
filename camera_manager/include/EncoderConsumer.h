#pragma once
extern "C" {
    #include <libavcodec/avcodec.h>
    #include <libavutil/avutil.h>
    #include <libavutil/hwcontext.h>
    #include <libavformat/avformat.h>
}

#include <string>
#include <stdexcept>
#include <atomic>
#include <thread>
#include "config.h"
#include "Consumer.h"
#include "Channel.h"
#include <memory>



class EncoderConsumer : public Consumer {
public:
    EncoderConsumer(const std::string& codec_name, int width, int height, int fps = 25);
    virtual ~EncoderConsumer();
    AVCodecContext* GetCodecContext() const { return codec_ctx; }
    void setInputChannel(FrameChannel* channel);
    void setOutputChannel(PacketChannel* channel);
    void start();
    void stop();
    virtual void run() override; 
private:
    AVCodecContext* codec_ctx;
    const AVCodec* codec;
    FrameChannel* m_input_channel{nullptr};
    PacketChannel* m_output_channel{nullptr};
    std::atomic_bool is_created{false};
    int m_width;
    int m_height;
    int m_fps;
};