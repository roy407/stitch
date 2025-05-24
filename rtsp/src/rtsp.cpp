#include "rtsp.h"
#include <iostream>
#include <unistd.h>
#include <sys/wait.h>
#include <fcntl.h>
#include "safe_queue.hpp"
extern "C" {
    #include <libavformat/avformat.h>
    #include <libavcodec/avcodec.h>
    #include <libavutil/opt.h>
    #include <libavutil/log.h>
}

rtsp_server::rtsp_server(safe_queue<AVPacket*>& packet_queues) : packet_queues(packet_queues) {
    running.store(false);
    codecpar = nullptr; 
    time_base = nullptr;
}

void rtsp_server::start_rtsp_server(AVCodecParameters** codecpar, AVRational* time_base, const std::string& push_stream_url) {
    if (running.load()) return;
    this->codecpar = codecpar;
    this->time_base = time_base;
    this->output_url = push_stream_url;
    running.store(true);
    t_rtsp = std::thread(&rtsp_server::push_stream, this); 
}

void rtsp_server::close_rtsp_server() {
    running.store(false);
}

bool rtsp_server::init_mediamtx() {
    auto getExecutableDir = [&]() -> std::string {
        char buf[PATH_MAX];
        ssize_t len = readlink("/proc/self/exe", buf, sizeof(buf)-1);
        if (len == -1) {
            std::cerr << "读取程序路径失败: " << strerror(errno) << std::endl;
            return "";
        }
        buf[len] = '\0';
        std::string exePath(buf);
        size_t lastSlash = exePath.find_last_of('/');
        return (lastSlash != std::string::npos) ? exePath.substr(0, lastSlash) : "";
    };
    std::string exeDir = getExecutableDir();
    if (exeDir.empty()) {
        std::cerr << "错误：无法获取程序路径！" << std::endl;
        return EXIT_FAILURE;
    }
    std::string childPath = exeDir + "/mediamtx/mediamtx";
    signal(SIGCHLD, SIG_IGN);
    pid = fork();
    if (pid == -1) {
        std::cerr << "fork失败: " << strerror(errno) << std::endl;
        return EXIT_FAILURE;
    } else if (pid == 0) {
        execl(childPath.c_str(), "mediamtx", nullptr);
        std::cerr << "执行失败: " << strerror(errno) << std::endl;
        _exit(EXIT_FAILURE);
    }
    std::cout << "成功启动子进程 PID: " << pid << std::endl;
    return EXIT_SUCCESS;
}

void rtsp_server::push_stream() {
    if (!*codecpar || (*codecpar)->codec_type != AVMEDIA_TYPE_VIDEO) {
        std::cout << "Invalid codec parameters" << std::endl;
        return;
    }
    AVFormatContext* out_ctx = nullptr;
    avformat_alloc_output_context2(&out_ctx, nullptr, "rtsp", output_url.c_str());
    AVStream* out_stream = avformat_new_stream(out_ctx, nullptr);
    avcodec_parameters_copy(out_stream->codecpar, *codecpar);
    out_stream->time_base = *time_base;
    av_opt_set(out_ctx->priv_data, "rtsp_transport", "tcp", 0);
    av_opt_set(out_ctx->priv_data, "muxdelay", "0.1", 0);
    int ret1 = avformat_write_header(out_ctx, NULL);
    while(running) {
        AVPacket* pkt;
        if(!packet_queues.try_pop(pkt)) continue;
        av_interleaved_write_frame(out_ctx, pkt);
        av_packet_unref(pkt);
    }
}

rtsp_server::~rtsp_server() {
    close_rtsp_server();
    if (t_rtsp.joinable()) {
        t_rtsp.join();
    }
    std::cout<<__func__<<" exit!"<<std::endl;
}

pid_t rtsp_server::pid = 0;