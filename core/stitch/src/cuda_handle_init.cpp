#include "cuda_handle_init.h"

cuda_handle_init::cuda_handle_init() {
    hw_device_ctx = nullptr;
    int ret = av_hwdevice_ctx_create(&hw_device_ctx, AV_HWDEVICE_TYPE_ASCEND, nullptr, nullptr, 0);
    if (ret < 0) {
        char errbuf[128];
        av_strerror(ret, errbuf, sizeof(errbuf));
        std::cerr << "Failed to create CUDA device context: " << errbuf << std::endl;
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
    static cuda_handle_init k;
    return k.hw_device_ctx;
}
