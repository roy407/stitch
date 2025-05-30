#include "cuda_handle_init.h"

cuda_handle_init::cuda_handle_init() {
    hw_device_ctx = nullptr;
    if (av_hwdevice_ctx_create(&hw_device_ctx, AV_HWDEVICE_TYPE_CUDA, nullptr, nullptr, 0) < 0) {
        throw std::runtime_error("Failed to create CUDA device context");
    }
}
cuda_handle_init::~cuda_handle_init() {
    av_buffer_unref(&hw_device_ctx);
}

AVBufferRef* cuda_handle_init::GetGPUDeviceHandle() {
    static cuda_handle_init k;
    return k.hw_device_ctx;
}
