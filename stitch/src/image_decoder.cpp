
#include "image_decoder.h"
#include <vector>

image_decoder::image_decoder(const std::string& codec_name) {
    codec = avcodec_find_decoder_by_name("h264_cuvid");
    if (!codec) {
        throw std::runtime_error("CUDA decoder not found: " + codec_name);
    }

    codec_ctx = avcodec_alloc_context3(codec);
    if (!codec_ctx) {
        throw std::runtime_error("Could not allocate codec context");
    }

    AVBufferRef* hw_device_ctx = nullptr;
    if (av_hwdevice_ctx_create(&hw_device_ctx, AV_HWDEVICE_TYPE_CUDA, nullptr, nullptr, 0) < 0) {
        throw std::runtime_error("Failed to create CUDA device context");
    }
    codec_ctx->hw_device_ctx = av_buffer_ref(hw_device_ctx);
    is_created.store(false);
}

image_decoder::~image_decoder() {
    avcodec_free_context(&codec_ctx);
    std::cout<<__func__<<" exit!"<<std::endl;
}

void image_decoder::set_parameter(AVCodecParameters* codecpar) {
    if(!is_created) {
        avcodec_parameters_to_context(codec_ctx, codecpar);
        if (avcodec_open2(codec_ctx, codec, nullptr) < 0) {
            throw std::runtime_error("Failed to open codec");
        }
        is_created.store(true);
    }
}

std::queue<AVFrame*> image_decoder::do_decode(const AVPacket* pkt) {
    std::queue<AVFrame*> q;
    int ret = avcodec_send_packet(codec_ctx, pkt);
    if (ret < 0) {
        char errbuf[256];
        av_strerror(ret, errbuf, sizeof(errbuf));
        std::cerr << "avcodec_send_packet error: " << errbuf << std::endl;
        return q;
    }

    while (ret >= 0) {
        frame = av_frame_alloc();
        if (!frame) {
            throw std::runtime_error("Could not allocate AVFrame");
        }
        ret = avcodec_receive_frame(codec_ctx, frame);
        if (ret == 0) {
            if (frame->format == AV_PIX_FMT_CUDA) {
                q.push(frame);
            }
        } else if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
            break;
        } else {
            char errbuf[256];
            av_strerror(ret, errbuf, sizeof(errbuf));
            std::cerr << "avcodec_receive_frame error: " << errbuf << std::endl;
            break;
        }
    }
    return q;
}
