#pragma once
#include <thread>
#include <atomic>

extern "C" {
    #include <libavformat/avformat.h>
    #include <libavcodec/avcodec.h>
    #include <libavutil/opt.h>
    #include <libavutil/log.h>
}


class rtsp_server : public std::thread {
public:
    explicit rtsp_server(AVCodecParameters** codecpar, AVRational* time_base, AVPacket** packet_queues, const std::string& push_stream_url)
        : codecpar(codecpar),time_base(time_base),packet_queues(packet_queues),output_url(push_stream_url),std::thread(&rtsp_server::push_stream, this){
    }
    static bool init_server();
    static bool close_server();
    ~rtsp_server();
private:
    void push_stream();
private:
    std::atomic_bool running;
    AVCodecParameters** codecpar; 
    AVRational* time_base;
    AVPacket** packet_queues;
    const std::string& output_url;
    static pid_t child_pid;
};