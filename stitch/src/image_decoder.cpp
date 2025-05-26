
#include "image_decoder.h"
#include <vector>
#include <thread>

image_decoder::image_decoder(safe_queue<AVPacket*>& in_packet , safe_queue<AVFrame*>& out_frame, const std::string& codec_name) : in_packet(in_packet), out_frame(out_frame) {
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
    running.store(false);
}

image_decoder::~image_decoder() {
    avcodec_free_context(&codec_ctx);
    close_image_decoder();
    if(t_img_decoder.joinable()) {
        t_img_decoder.join();
    }
    std::cout<<__func__<<" exit!"<<std::endl;
}

void image_decoder::start_image_decoder(AVCodecParameters* codecpar) {
    if(!is_created) {
        avcodec_parameters_to_context(codec_ctx, codecpar);
        if (avcodec_open2(codec_ctx, codec, nullptr) < 0) {
            throw std::runtime_error("Failed to open codec");
        }
        is_created.store(true);
        running.store(true);
        t_img_decoder = std::thread(&image_decoder::do_decode,this);
    }
}

void image_decoder::close_image_decoder() {
    running.store(false);
}

void image_decoder::do_decode() {
    AVPacket* pkt = nullptr;
    while(running) {
        if(!in_packet.try_pop(pkt)) continue;
        int ret = avcodec_send_packet(codec_ctx, pkt);
        if (ret < 0) {
            char errbuf[256];
            av_strerror(ret, errbuf, sizeof(errbuf));
            std::cerr << "avcodec_send_packet error: " << errbuf << std::endl;
            return;
        }

        while (ret >= 0) {
            AVFrame* frame = av_frame_alloc();
            if (!frame) {
                throw std::runtime_error("Could not allocate AVFrame");
            }
            ret = avcodec_receive_frame(codec_ctx, frame);
            if (ret == 0) {
                if (frame->format == AV_PIX_FMT_CUDA) {
                    out_frame.push(frame);
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
        av_packet_unref(pkt);
        av_packet_free(&pkt); //堆中内存，需释放
    }
}
