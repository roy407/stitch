#pragma once

#include <atomic>
#include <stdexcept>

extern "C" {
    #include <libavcodec/avcodec.h>
    #include <libavformat/avformat.h>
    #include <libavutil/avutil.h>
    #include <libavutil/hwcontext.h>
}

class cuda_handle_init {
public:
    ~cuda_handle_init();
    static AVBufferRef* GetGPUDeviceHandle();
private:
    cuda_handle_init();
private:
    AVBufferRef* hw_device_ctx;
};