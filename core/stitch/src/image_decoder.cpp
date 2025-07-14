
#include "image_decoder.h"
#include "cuda_handle_init.h"
#include <vector>
#include <thread>
#include <chrono>
#include "tools.hpp"

image_decoder::image_decoder(safe_queue<std::pair<AVPacket*,costTimes>>& packet_input , safe_queue<std::pair<AVFrame*,costTimes>>& frame_output, int cam_id, const std::string& codec_name) : packet_input(packet_input), frame_output(frame_output), cam_id(cam_id) {
    
    codec = avcodec_find_decoder_by_name("h264_cuvid");
    if (!codec) {
        throw std::runtime_error("CUDA decoder not found: " + codec_name);
    }

    codec_ctx = avcodec_alloc_context3(codec);
    if (!codec_ctx) {
        throw std::runtime_error("Could not allocate codec context");
    }

    codec_ctx->hw_device_ctx = av_buffer_ref(cuda_handle_init::GetGPUDeviceHandle());

    is_created.store(false);
    running.store(false);
}

image_decoder::~image_decoder() {
    close_image_decoder();
    if(t_img_decoder.joinable()) {
        t_img_decoder.join();
    }
    if(codec_ctx && codec_ctx->hw_device_ctx) {
        av_buffer_unref(&(codec_ctx->hw_device_ctx));
        codec_ctx->hw_device_ctx = nullptr;
    }
    avcodec_free_context(&codec_ctx);
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
    std::pair<AVPacket*,costTimes> pkt;
    while(running) {
        packet_input.wait_and_pop(pkt);
        int ret = avcodec_send_packet(codec_ctx, pkt.first);
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
                    pkt.second.when_get_decoded_frame[cam_id] = get_now_time();
                    frame_output.push({frame, pkt.second});
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
        av_packet_unref(pkt.first);
    }
}
