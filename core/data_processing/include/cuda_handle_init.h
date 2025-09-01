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

class cuda_handle_init {
public:
    ~cuda_handle_init();
    static AVBufferRef* GetGPUDeviceHandle();
private:
    cuda_handle_init();
private:
    AVBufferRef* hw_device_ctx;
};