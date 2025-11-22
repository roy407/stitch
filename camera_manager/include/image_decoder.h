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
#include "TaskManager.h"

class image_decoder : TaskManager {
public:
    image_decoder(const std::string& codec_name = "h264_cuvid");
    ~image_decoder();
    void start_image_decoder(int cam_id, AVCodecParameters* codecpar, safe_queue<Frame>* m_frame, safe_queue<Packet>* m_packet);
    void start_image_decoder(int cam_id, AVCodecParameters* codecpar, std::vector<safe_queue<Frame>*> m_frames, safe_queue<Packet>* m_packet);
    void close_image_decoder();
    virtual void run();
private:
    int cam_id;
    AVCodecContext* codec_ctx;
    const AVCodec* codec;
    std::vector<safe_queue<Frame>*> m_frameOutput;
    safe_queue<Packet>* m_packetInput;
};
