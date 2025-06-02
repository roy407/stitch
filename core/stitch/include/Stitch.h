#pragma once
#include <thread>
#include <atomic>
extern "C" {
    #include <libavformat/avformat.h>
    #include <libavcodec/avcodec.h>
    #include <libavutil/opt.h>
    #include <libavutil/log.h>
}
class Stitch {
public:
    explicit Stitch(int width,int height,int cam_num);
    ~Stitch();
    AVFrame* do_stitch(AVFrame** inputs);
private:
    AVFrame* output;
    AVBufferRef* hw_frames_ctx;
    std::atomic_bool running;
    const int cam_num;
    const int single_width;
    const int height;
};