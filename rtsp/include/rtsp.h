#pragma once
#include <thread>
#include <atomic>

extern "C" {
    #include <libavformat/avformat.h>
    #include <libavcodec/avcodec.h>
    #include <libavutil/opt.h>
    #include <libavutil/log.h>
}

#include "safe_queue.hpp"

class rtsp_server {
public:
    rtsp_server(safe_queue<AVPacket*>& packet_queues);
    void start_rtsp_server(AVCodecParameters** codecpar, AVRational* time_base, const std::string& push_stream_url);
    void close_rtsp_server();
    static bool init_mediamtx();
    ~rtsp_server();
private:
    void push_stream();
private:
    std::atomic_bool running;
    AVCodecParameters** codecpar; 
    AVRational* time_base;
    safe_queue<AVPacket*>& packet_queues;
    std::string output_url;
    static pid_t pid;
    std::thread t_rtsp;
};