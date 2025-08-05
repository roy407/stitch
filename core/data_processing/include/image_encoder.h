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
#include "safe_queue.hpp"
#include "tools.hpp"

class image_encoder {
public:
    image_encoder(int width, int height, safe_queue<std::pair<AVFrame*,costTimes>>& frame_input,safe_queue<std::pair<AVPacket*,costTimes>>& packet_output, const std::string& codec_name = "h264_ascend");
    ~image_encoder();
    void start_image_encoder();
    void close_image_encoder();

    void do_encode();

    AVCodecContext* codec_ctx;
    const AVCodec* codec;
    AVPacket* pkt;
    safe_queue<std::pair<AVFrame*,costTimes>>& frame_input;
    safe_queue<std::pair<AVPacket*,costTimes>>& packet_output;
    std::atomic_bool is_created;
    std::atomic_bool running;
    std::thread t_img_encoder;
    int width;
    int height;
};