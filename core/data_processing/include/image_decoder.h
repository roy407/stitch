#pragma once
extern "C" {
    #include <libavcodec/avcodec.h>
    #include <libavutil/avutil.h>
    #include <libavutil/hwcontext.h>
    #include <libavformat/avformat.h>
}

#include <iostream>
#include "safe_queue.hpp"
#include <stdexcept>
#include <atomic>
#include <thread>
#include <memory>

class image_decoder {
public:
    image_decoder(const std::string& codec_name = "h264_cuvid");
    ~image_decoder();
    void start_image_decoder(AVCodecParameters* codecpar, safe_queue<AVFrame*>* m_frame, safe_queue<AVPacket*>* m_packet);
    void close_image_decoder();
    void do_decode();
private:
    AVCodecContext* codec_ctx;
    const AVCodec* codec;
    safe_queue<AVFrame*>* m_frameOutput;
    safe_queue<AVPacket*>* m_packetInput;
    std::thread m_thread;
    bool running{false};
};
