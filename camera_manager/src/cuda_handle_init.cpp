#include "cuda_handle_init.h"
#include "log.hpp"

cuda_handle_init::cuda_handle_init() {
    hw_device_ctx = nullptr;
    LOG_DEBUG("av_hwdevice_ctx_create start");
    //耗时近2s
    int ret = av_hwdevice_ctx_create(&hw_device_ctx, AV_HWDEVICE_TYPE_CUDA, nullptr, nullptr, 0);
    LOG_DEBUG("av_hwdevice_ctx_create over");
    if (ret < 0) {
        char errbuf[128];
        av_strerror(ret, errbuf, sizeof(errbuf));
        LOG_ERROR("Failed to create CUDA device context: {}" ,errbuf);
        throw std::runtime_error("Failed to create CUDA device context");
    }
}
cuda_handle_init::~cuda_handle_init() {
    if (hw_device_ctx) {
        av_buffer_unref(&hw_device_ctx);
        hw_device_ctx = nullptr;
    }
}

AVBufferRef* cuda_handle_init::GetGPUDeviceHandle() {
    LOG_DEBUG("GetGPUDeviceHandle");
    static cuda_handle_init k;
    return k.hw_device_ctx;
}
