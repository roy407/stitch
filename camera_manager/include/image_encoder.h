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
#include "TaskManager.h"
#include "config.h"
class image_encoder : public TaskManager {
public:
    image_encoder(int width, int height, safe_queue<Frame>& frame_input,safe_queue<Packet>& packet_output, const std::string& codec_name = config::GetInstance().GetGlobalConfig().image_encoder);
    ~image_encoder();
    void start_image_encoder();
    void close_image_encoder();

    void run();

    AVCodecContext* codec_ctx;
    const AVCodec* codec;
    AVPacket* pkt;
    safe_queue<Frame>& frame_input;
    safe_queue<Packet>& packet_output;
    std::atomic_bool is_created;
    int width;
    int height;
};