
#include "DecoderConsumer.h"
#include "cuda_handle_init.h"
#include <vector>
#include <thread>
#include <chrono>
#include "log.hpp"
extern "C" {
    #include <libavutil/imgutils.h>
    // 其他FFmpeg头文件，如 libavcodec/avcodec.h, libavformat/avformat.h 等
}

DecoderConsumer::DecoderConsumer(const std::string& codec_name) {
    m_name += codec_name + "_decoder";
    codec = avcodec_find_decoder_by_name(codec_name.c_str());
    if (!codec) {
        LOG_ERROR("Codec {} not found", codec_name);
    }

    codec_ctx = avcodec_alloc_context3(codec);
    if (!codec_ctx) {
        LOG_ERROR("Could not allocate codec context");
    }

    codec_ctx->hw_device_ctx = av_buffer_ref(cuda_handle_init::GetGPUDeviceHandle());

    m_channel2resize = new FrameChannel;
    m_channel2stitch = new FrameChannel;
}

DecoderConsumer::~DecoderConsumer() {
    delete m_channel2resize;
    delete m_channel2stitch;

    if(codec_ctx && codec_ctx->hw_device_ctx) {
        av_buffer_unref(&(codec_ctx->hw_device_ctx));
        codec_ctx->hw_device_ctx = nullptr;
    }
    avcodec_free_context(&codec_ctx);
}

void DecoderConsumer::setAVCodecParameters(AVCodecParameters *codecpar, AVRational time_base) {
    if (codec_ctx== nullptr) {
        throw std::runtime_error("Codec context is not initialized");
    }
    if (codecpar== nullptr) {
        throw std::runtime_error("AVCodecParameters is null");
    }
    avcodec_parameters_to_context(codec_ctx, codecpar);
    codec_ctx->time_base = time_base;
    codec_ctx->pkt_timebase = time_base;
}

void DecoderConsumer::setChannel(PacketChannel *channel) {
    if (channel == nullptr) {
        throw std::runtime_error("PacketChannel is null");
    }
    m_channelFromAVFramePro = channel;
}

FrameChannel *DecoderConsumer::getChannel2Resize() {
    return m_channel2resize;
}

FrameChannel *DecoderConsumer::getChannel2Stitch() {
    return m_channel2stitch;
}

void DecoderConsumer::start() {
    if(m_channelFromAVFramePro == nullptr) {
        LOG_ERROR("m_channelFromAVFramePro has not inited");
        return;
    }
    
    if (avcodec_open2(codec_ctx, codec, nullptr) < 0) {
        throw std::runtime_error("Failed to open codec");
    }
    TaskManager::start();
}

void DecoderConsumer::stop() {
    m_channelFromAVFramePro->stop();
    TaskManager::stop();
}

void DecoderConsumer::run() {
    Packet pkt;
    // int src_width = 3840;
    // int src_height = 2160;
    if(!m_channelFromAVFramePro) throw std::runtime_error("null pointer");
    if(!m_channel2stitch) throw std::runtime_error("null pointer");
    // AVBufferRef *hw_frames_ctx = av_hwframe_ctx_alloc(cuda_handle_init::GetGPUDeviceHandle());
    // AVHWFramesContext *frames_ctx = (AVHWFramesContext*)hw_frames_ctx->data;
    // frames_ctx->format = AV_PIX_FMT_CUDA;
    // frames_ctx->sw_format = AV_PIX_FMT_YUV420P;
    // frames_ctx->width = src_width;
    // frames_ctx->height = src_height;
    // av_hwframe_ctx_init(hw_frames_ctx);
    while(running) {
        if(!m_channelFromAVFramePro->recv(pkt)) break;
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
                    frame.cam_id = pkt.cam_id;
                    frame.m_costTimes = pkt.m_costTimes;
                    frame.m_costTimes.when_get_decoded_frame[frame.cam_id] = get_now_time();
                    frame.m_timestamp = pkt.m_timestamp; // 将packet时间戳提供给frame
                    m_channel2stitch->send(frame);
                    
                    Frame frame_copy;
                    frame_copy.cam_id = frame.cam_id;
                    frame_copy.m_costTimes = frame.m_costTimes;
                    frame_copy.m_data = av_frame_alloc();
                    int ret = av_frame_ref(frame_copy.m_data, frame.m_data);
                    m_channel2resize->send(frame_copy);
                // } else if (frame.m_data->format == AV_PIX_FMT_YUV420P) {
                //     frame.cam_id = pkt.cam_id;
                //     frame.m_costTimes = pkt.m_costTimes;
                //     frame.m_costTimes.when_get_decoded_frame[frame.cam_id] = get_now_time();
                //     frame.m_timestamp = pkt.m_timestamp; // 将packet时间戳提供给frame
                //     AVFrame *hw_frame = av_frame_alloc();
                //     AVFrame *cpu_frame = frame.m_data;
                //     hw_frame->hw_frames_ctx = av_buffer_ref(hw_frames_ctx);
                //     hw_frame->width = cpu_frame->width;
                //     hw_frame->height = cpu_frame->height;
                //     hw_frame->format = AV_PIX_FMT_CUDA;
                //     av_hwframe_get_buffer(hw_frame->hw_frames_ctx, hw_frame, 0);

                //     // 执行传输
                //     if (av_hwframe_transfer_data(hw_frame, cpu_frame, 0) == 0) {
                //         av_frame_free(&cpu_frame);
                //         frame.m_data = hw_frame;
                //         frame.m_costTimes.when_get_decoded_frame[frame.cam_id] = get_now_time();
                //         m_channel2stitch->send(frame);
                        
                //         Frame frame_copy;
                //         frame_copy.cam_id = frame.cam_id;
                //         frame_copy.m_costTimes = frame.m_costTimes;
                //         frame_copy.m_data = av_frame_alloc();
                //         int ret = av_frame_ref(frame_copy.m_data, frame.m_data);
                //         m_channel2resize->send(frame_copy);
                //     }
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
        av_packet_free(&pkt.m_data);
    }
    m_channelFromAVFramePro->clear();
}
