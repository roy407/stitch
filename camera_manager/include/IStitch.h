#pragma once
#include <thread>
#include <atomic>
#include <cuda_runtime.h>
extern "C" {
    #include <libavformat/avformat.h>
    #include <libavcodec/avcodec.h>
    #include <libavutil/opt.h>
    #include <libavutil/log.h>
}
#include "cuda_handle_init.h"
#include "config.h"

template<int FormatValue>
struct Format {
    static constexpr int value = FormatValue;
    /*
    0: YUV420
    1: YUV420P
    */
};

using YUV420 = Format<0>;
using YUV420P = Format<1>;

struct RawKernel {};
struct CropKernel {};
struct HMatrixInvKernel {};
struct HMatrixInvV1_1Kernel {};
struct HMatrixInvV2Kernel {};
struct MappingTableKernel {};

template <typename Derived>
class IStitch {
public:
    ~IStitch() {
        if (hw_frames_ctx) {
            av_buffer_unref(&hw_frames_ctx);
        }
    };

    void init(int num, int single_width, int output_width, int height) {
        this->num = num;
        this->single_width = single_width;
        this->output_width = output_width;
        this->height = height;
        LOG_DEBUG("stitch initing ... cam_num : {} , width : {} , stitched_width : {} , height : {}", num, single_width, output_width, height);
        CreateHWFramesCtx();
        static_cast<Derived*>(this)->init_impl();
    }

    AVFrame* do_stitch(AVFrame** inputs) {
        return static_cast<Derived*>(this)->do_stitch_impl(inputs);
    }

protected:
    AVFrame* output{nullptr};
    AVBufferRef* hw_frames_ctx{nullptr};

    int num = 0;
    int single_width = 0;
    int output_width = 0;
    int height = 0;

    bool CreateHWFramesCtx();
};

// 创建 HW frame context
template <typename Derived>
bool IStitch<Derived>::CreateHWFramesCtx() {
    hw_frames_ctx = av_hwframe_ctx_alloc(cuda_handle_init::GetGPUDeviceHandle());
    AVHWFramesContext* frames_ctx = (AVHWFramesContext*)hw_frames_ctx->data;
    frames_ctx->format = AV_PIX_FMT_CUDA;
    frames_ctx->sw_format = AV_PIX_FMT_NV12;   // CUDA 支持的底层格式
    frames_ctx->width = output_width;
    frames_ctx->height = height;
    frames_ctx->initial_pool_size = 20;

    if (av_hwframe_ctx_init(hw_frames_ctx) < 0) {
        throw std::runtime_error("Failed to initialize CUDA hwframe context");
    }
    return true;
}