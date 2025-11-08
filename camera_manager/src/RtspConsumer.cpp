#include "RtspConsumer.h"
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
#include "log.hpp"

RtspConsumer::RtspConsumer(safe_queue<Packet>& packet, AVCodecParameters** codecpar, AVRational* time_base, const std::string& push_stream_url) : packet_input(packet) {
    m_name += "rtsp";
    this->codecpar = codecpar;
    this->time_base = time_base;
    this->output_url = push_stream_url;
    if (!*codecpar || (*codecpar)->codec_type != AVMEDIA_TYPE_VIDEO) {
        LOG_ERROR("Invalid codec parameters");
        return;
    }
    avformat_alloc_output_context2(&out_ctx, nullptr, "rtsp", output_url.c_str());
    AVStream* out_stream = avformat_new_stream(out_ctx, nullptr);
    avcodec_parameters_copy(out_stream->codecpar, *codecpar);
    out_stream->time_base = *time_base;
    av_opt_set(out_ctx->priv_data, "rtsp_transport", "tcp", 0);
    av_opt_set(out_ctx->priv_data, "muxdelay", "0.1", 0);
    int ret1 = avformat_write_header(out_ctx, NULL);
}
void RtspConsumer::start() {
    TaskManager::start();
}
void RtspConsumer::stop() {
    TaskManager::stop();
}

void RtspConsumer::run() {
    while(running) {
        Packet pkt;
        if(!packet_input.wait_and_pop(pkt)) break;
        int ret = av_interleaved_write_frame(out_ctx, pkt.m_data);
        av_packet_unref(pkt.m_data);
    }
    packet_input.clear();
}

bool RtspConsumer::init_mediamtx() {
    auto getExecutableDir = [&]() -> std::string {
        char buf[PATH_MAX];
        ssize_t len = readlink("/proc/self/exe", buf, sizeof(buf)-1);
        if (len == -1) {
            LOG_ERROR("读取程序路径失败: {}",strerror(errno));
            return "";
        }
        buf[len] = '\0';
        std::string exePath(buf);
        size_t lastSlash = exePath.find_last_of('/');
        return (lastSlash != std::string::npos) ? exePath.substr(0, lastSlash) : "";
    };
    std::string exeDir = getExecutableDir();
    if (exeDir.empty()) {
        LOG_ERROR("错误：无法获取程序路径！");
        return EXIT_FAILURE;
    }
    std::string childPath = exeDir + "/mediamtx/mediamtx";
    signal(SIGCHLD, SIG_IGN);
    pid = fork();
    if (pid == -1) {
        LOG_ERROR("fork失败: {}",strerror(errno));
        return EXIT_FAILURE;
    } else if (pid == 0) {
        execl(childPath.c_str(), "mediamtx", nullptr);
        LOG_ERROR("执行失败: {}",strerror(errno));
        _exit(EXIT_FAILURE);
    }
    LOG_INFO("成功启动子进程 PID: {}",pid);
    return EXIT_SUCCESS;
}

bool RtspConsumer::destory_mediamtx() {
    int pid = 0;
    std::string cmd = "pgrep mediamtx";
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) {
        LOG_ERROR("Failed to run pgrep command");
        return false;
    }

    char buffer[128];
    while (fgets(buffer, sizeof(buffer), pipe)) {
        int tmp_pid = std::atoi(buffer);
        if(pid > 0) break;
        if (tmp_pid > 0) {
            pid = tmp_pid;
        }
    }
    pclose(pipe);

    if (kill(pid, SIGTERM) == 0) {
        LOG_INFO("Successfully sent SIGTERM to PID {}",pid);
    } else {
        LOG_ERROR("Failed to kill PID {}", pid);
    }
    return true;
}

RtspConsumer::~RtspConsumer() {
    LOG_DEBUG("{} exit!",__func__);
}

pid_t RtspConsumer::pid = 0;