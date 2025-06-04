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

class image_decoder {
public:
    image_decoder(safe_queue<AVPacket*>& packet_input, safe_queue<AVFrame*>& frame_output, const std::string& codec_name = "h264_cuvid");
    ~image_decoder();
    void start_image_decoder(AVCodecParameters* codecpar);
    void close_image_decoder();
    void do_decode();

    AVCodecContext* codec_ctx;
    const AVCodec* codec;
    safe_queue<AVPacket*>& packet_input;
    safe_queue<AVFrame*>& frame_output;
    std::atomic_bool is_created;
    std::atomic_bool running;
    std::thread t_img_decoder;
};
