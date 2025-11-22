#pragma once
#include "Consumer.h"
extern "C" {
    #include "libavformat/avformat.h"
    #include "libavcodec/avcodec.h"
    #include "libavutil/pixfmt.h" 
    #include "libavutil/pixdesc.h" 
    #include "libavutil/opt.h"
    #include "libavutil/log.h"
    #include "libavcodec/bsf.h"
}
#include "safe_queue.hpp"

class ResizeConsumer : public Consumer {
public:
    ResizeConsumer(int width, int height, float scale_factor);
    ResizeConsumer(int width, int height, AVRational rational);
    ResizeConsumer(int width, int height, int output_width, int output_height);
    void SetInputFrame(safe_queue<Frame>* InputFrame);
    virtual void start();
    virtual void stop();
    virtual void run();
    virtual ~ResizeConsumer();
    safe_queue<Frame>& get_resize_frame();
private:
    AVBufferRef* hw_frames_ctx{nullptr};
    int width{0};
    int height{0};
    int output_width{0};
    int output_height{0};
    safe_queue<Frame>* InputFrame{nullptr};
    safe_queue<Frame> OutputFrame;
};