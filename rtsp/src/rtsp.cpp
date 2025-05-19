#include "rtsp.h"
#include <iostream>
#include <unistd.h>
#include <sys/wait.h>
#include <fcntl.h>
extern "C" {
    #include <libavformat/avformat.h>
    #include <libavcodec/avcodec.h>
    #include <libavutil/opt.h>
    #include <libavutil/log.h>
}

bool rtsp_server::init_server() {
    pid_t pid = fork();
    if (pid < 0) {
        perror("fork failed");
        return false;
    }

    if (pid == 0) {
        int null_fd = open("/dev/null", O_RDWR);
        if (null_fd == -1) {
            perror("open /dev/null failed");
            exit(EXIT_FAILURE);
        }
        dup2(null_fd, STDIN_FILENO);
        dup2(null_fd, STDOUT_FILENO);
        dup2(null_fd, STDERR_FILENO);
        close(null_fd);
        execlp("mediamtx/mediamtx",
               "mediamtx",
               (char*)NULL);
        perror("execlp mediamtx failed");
        exit(EXIT_FAILURE);
    }

    // 父进程记录PID
    child_pid = pid;
    return true;
}

bool rtsp_server::close_server() {
    if (child_pid > 0) {
        kill(child_pid, SIGTERM);
        waitpid(child_pid, NULL, 0);
        child_pid = 0;
    }
}

void rtsp_server::push_stream() {
    if (!*codecpar || (*codecpar)->codec_type != AVMEDIA_TYPE_VIDEO) {
        std::cout << "Invalid codec parameters" << std::endl;
        return;
    }
    running.store(true);
    AVFormatContext* out_ctx = nullptr;
    avformat_alloc_output_context2(&out_ctx, nullptr, "rtsp", output_url.c_str());
    AVStream* out_stream = avformat_new_stream(out_ctx, nullptr);
    avcodec_parameters_copy(out_stream->codecpar, *codecpar);
    out_stream->time_base = *time_base;
    av_opt_set(out_ctx->priv_data, "rtsp_transport", "tcp", 0);
    av_opt_set(out_ctx->priv_data, "muxdelay", "0.1", 0);
    int ret1 = avformat_write_header(out_ctx, nullptr);
    while(running) {
        AVPacket* pkt = *packet_queues;
        if(pkt == nullptr) continue;
        pkt->pts = av_rescale_q(pkt->pts, *time_base, out_stream->time_base);
        pkt->dts = av_rescale_q(pkt->dts, *time_base, out_stream->time_base);
        av_interleaved_write_frame(out_ctx, pkt);
        av_packet_unref(pkt);
    }
}

rtsp_server::~rtsp_server() {
    running.store(false);
    if (this->joinable()) {
        this->join();
    }
}

pid_t rtsp_server::child_pid = 0;