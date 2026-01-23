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
// #include "safe_queue.hpp"
#include "list_queue.hpp"
#include "tools.hpp"
#include "TaskManager.h"
#include "config.h"

// 待修改
class EncoderConsumer : public TaskManager {
public:
    //EncoderConsumer(int width, int height, safe_queue<Frame>& frame_input,safe_queue<Packet>& packet_output, const std::string& codec_name = CFG_HANDLE.GetGlobalConfig().encoder);
    EncoderConsumer(int width, int height, list_queue<Frame>& frame_input,list_queue<Packet>& packet_output, const std::string& codec_name = CFG_HANDLE.GetGlobalConfig().encoder);
    virtual ~EncoderConsumer();
    void start_image_encoder();
    void close_image_encoder();

    void run();

    AVCodecContext* codec_ctx;
    const AVCodec* codec;
    AVPacket* pkt;
    //safe_queue<Frame>& frame_input;
    //safe_queue<Packet>& packet_output;
    list_queue<Frame>& frame_input;
    list_queue<Packet>& packet_output;
    std::atomic_bool is_created;
    int width;
    int height;
};