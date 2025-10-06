
#include "image_decoder.h"
#include "cuda_handle_init.h"
#include <vector>
#include <thread>
#include <chrono>
#include "log.hpp"

image_decoder::image_decoder(const std::string& codec_name) {
    codec = avcodec_find_decoder_by_name(codec_name.c_str());
    if (!codec) {
        throw std::runtime_error("CUDA decoder not found: " + codec_name);
    }

    codec_ctx = avcodec_alloc_context3(codec);
    if (!codec_ctx) {
        throw std::runtime_error("Could not allocate codec context");
    }

    codec_ctx->hw_device_ctx = av_buffer_ref(cuda_handle_init::GetGPUDeviceHandle());
}

image_decoder::~image_decoder() {
}

void image_decoder::start_image_decoder(int cam_id, AVCodecParameters* codecpar, safe_queue<Frame>* m_frame, safe_queue<Packet>* m_packet) {
    this->cam_id = cam_id;
    avcodec_parameters_to_context(codec_ctx, codecpar);
    if (avcodec_open2(codec_ctx, codec, nullptr) < 0) {
        throw std::runtime_error("Failed to open codec");
    }
    m_frameOutput = m_frame;
    m_packetInput = m_packet;
    running = true;
    m_thread = std::thread(&image_decoder::do_decode, this);
}

void image_decoder::close_image_decoder() {
    running = false;
    m_packetInput->stop();
    if(m_thread.joinable()) m_thread.join();
    if(codec_ctx && codec_ctx->hw_device_ctx) {
        av_buffer_unref(&(codec_ctx->hw_device_ctx));
        codec_ctx->hw_device_ctx = nullptr;
    }
    avcodec_free_context(&codec_ctx);
}

void image_decoder::do_decode() {
    Packet pkt;
    if(!m_packetInput) throw std::runtime_error("null pointer");
    if(!m_frameOutput) throw std::runtime_error("null pointer");
    while(running) {
        if(!m_packetInput->wait_and_pop(pkt)) break;
        int ret = avcodec_send_packet(codec_ctx, pkt.m_data);
        if (ret < 0) {
            char errbuf[256];
            av_strerror(ret, errbuf, sizeof(errbuf));
            LOG_ERROR("avcodec_send_packet error: {}",errbuf);
            return;
        }

        while (ret >= 0) {
            Frame frame;
            frame.m_data = av_frame_alloc();
            if (!frame.m_data) {
                throw std::runtime_error("Could not allocate AVFrame");
            }
            ret = avcodec_receive_frame(codec_ctx, frame.m_data);
            if (ret == 0) {
                if (frame.m_data->format == AV_PIX_FMT_CUDA) {
                    frame.m_costTimes = pkt.m_costTimes;
                    frame.m_costTimes.when_get_decoded_frame[cam_id] = get_now_time();
                    m_frameOutput->push(frame);
                }
            } else if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                break;
            } else {
                char errbuf[256];
                av_strerror(ret, errbuf, sizeof(errbuf));
                LOG_ERROR("avcodec_receive_frame error: {}", errbuf);
                break;
            }
        }
        av_packet_unref(pkt.m_data);
    }
    while(m_packetInput->size()) {
        m_packetInput->pop_and_free();
    }
    LOG_DEBUG("decoder thread exit!");
}
