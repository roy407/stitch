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

class cuda_init {
public:
    cuda_init();
    ~cuda_init();
    void start_cuda_init();
    void close_cuda_init();

    AVCodecContext* codec_ctx;
    static AVBufferRef* hw_device_ctx;
    const AVCodec* codec;
};