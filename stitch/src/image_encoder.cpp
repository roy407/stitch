#include "image_encoder.h"

extern "C" {
    #include <libavutil/opt.h>
}

image_encoder::image_encoder(const std::string& codec_name, int width, int height, int fps) {

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
    codec_ctx->pix_fmt = AV_PIX_FMT_CUDA; // 若使用 GPU 输入帧

    // 设置编码参数（示例）
    if (codec->id == AV_CODEC_ID_H264 || codec->id == AV_CODEC_ID_HEVC) {
        av_opt_set(codec_ctx->priv_data, "preset", "p1", 0); // p1=最快
    }

    // 打开编码器
    if (avcodec_open2(codec_ctx, codec, nullptr) < 0) {
        throw std::runtime_error("Failed to open encoder");
    }

    pkt = av_packet_alloc();
}

image_encoder::~image_encoder() {
    if (pkt) {
        av_packet_free(&pkt);
    }
    if (codec_ctx) {
        avcodec_free_context(&codec_ctx);
    }
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
