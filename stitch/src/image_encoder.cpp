#include "image_encoder.h"

extern "C" {
    #include <libavutil/opt.h>
}

image_encoder::image_encoder(const std::string& codec_name) {
    
    int width = 3840;
    int height = 2160;
    int fps = 20;
    
    codec = avcodec_find_encoder_by_name(codec_name.c_str());
    if (!codec) {
        throw std::runtime_error("Encoder not found: " + codec_name);
    }

    codec_ctx = avcodec_alloc_context3(codec);
    if (!codec_ctx) {
        throw std::runtime_error("Failed to allocate encoder context");
    }

    codec_ctx->width = width;
    codec_ctx->height = height;
    codec_ctx->time_base = AVRational{1, fps};
    codec_ctx->framerate = AVRational{fps, 1};
    codec_ctx->gop_size = 12;
    codec_ctx->max_b_frames = 0;
    codec_ctx->pix_fmt = AV_PIX_FMT_CUDA; 

    // Example for nvenc:
    if (codec->id == AV_CODEC_ID_H264 || codec->id == AV_CODEC_ID_HEVC) {
        av_opt_set(codec_ctx->priv_data, "preset", "p1", 0);
        av_opt_set(codec_ctx->priv_data, "gpu", "0", 0);
    }

    AVBufferRef* hw_device_ctx = nullptr;
    if (av_hwdevice_ctx_create(&hw_device_ctx, AV_HWDEVICE_TYPE_CUDA, nullptr, nullptr, 0) < 0) {
        throw std::runtime_error("Failed to create CUDA device context");
    }
    codec_ctx->hw_device_ctx = av_buffer_ref(hw_device_ctx);

    AVBufferRef* frames_ref = av_hwframe_ctx_alloc(hw_device_ctx);
    if (!frames_ref) {
        throw std::runtime_error("Failed to allocate HW frame context");
    }
    AVHWFramesContext* frames_ctx = (AVHWFramesContext*)(frames_ref->data);
    frames_ctx->format = AV_PIX_FMT_CUDA;
    frames_ctx->sw_format = AV_PIX_FMT_YUV420P;
    frames_ctx->width = width;
    frames_ctx->height = height;
    frames_ctx->initial_pool_size = 20;
    if (av_hwframe_ctx_init(frames_ref) < 0) {
        throw std::runtime_error("Failed to initialize HW frame context");
    }
    codec_ctx->hw_frames_ctx = av_buffer_ref(frames_ref);
}

image_encoder::~image_encoder() {
    close_image_encoder();
    if (pkt) {
        av_packet_free(&pkt);
    }
    if (codec_ctx && codec_ctx->hw_device_ctx) {
        av_buffer_unref(&codec_ctx->hw_device_ctx);
    }
    if (codec_ctx) {
        avcodec_free_context(&codec_ctx);
    }
}

void image_encoder::start_image_encoder() {
    if(!is_created) {
        if (avcodec_open2(codec_ctx, codec, nullptr) < 0) {
            throw std::runtime_error("Failed to open encoder");
        }
        is_created.store(true);
        pkt = av_packet_alloc();
    }
}
void image_encoder::close_image_encoder() {

}

AVPacket* image_encoder::do_encode(AVFrame* frame) {
    int ret = avcodec_send_frame(codec_ctx, frame);
    if (ret < 0) {
        throw std::runtime_error("Error sending frame to encoder");
    }

    ret = avcodec_receive_packet(codec_ctx, pkt);
    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
        return nullptr;  // 无输出可用
    } else if (ret < 0) {
        throw std::runtime_error("Error during encoding");
    }

    AVPacket* out_pkt = av_packet_alloc();
    av_packet_ref(out_pkt, pkt);
    av_packet_unref(pkt);
    return out_pkt;
}