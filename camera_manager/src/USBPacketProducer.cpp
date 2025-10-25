#include "USBPacketProducer.h"
#include <iostream>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/videodev2.h>
#include <sys/mman.h>
#include <cstring>

USBPacketProducer::USBPacketProducer(CameraConfig camera_config)
{
    this->cam_id = camera_config.cam_id;
    m_name += camera_config.name;
    std::string video_size = std::to_string(camera_config.width) + "x" + std::to_string(camera_config.height);
    fmt_ctx = avformat_alloc_context();
    av_dict_set(&options, "pixel_format", "mjpeg", 0);
    av_dict_set(&options, "video_size", video_size.c_str(), 0);
    av_dict_set(&options, "framerate", "30", 0);
    iformat = av_find_input_format("v4l2");
    if (!iformat) {
        LOG_ERROR("v4l2 input format not found");
        return;
    }
    cam_path = camera_config.input_url.c_str();
    m_status.width = camera_config.width;
    m_status.height = camera_config.height;
    LOG_DEBUG("try to open v4l2 device: [{}]", cam_path);
    int ret = avformat_open_input(&fmt_ctx, cam_path.c_str(), iformat, &options);
    if (ret < 0) {
        LOG_ERROR("open usb link {} failed", cam_path);
        char errbuf[256];
        av_strerror(ret, errbuf, sizeof(errbuf));
        LOG_ERROR("open_input failed: {}", errbuf);
        return;
    }
    ret = avformat_find_stream_info(fmt_ctx, nullptr);
    if (ret < 0) {
        LOG_ERROR("Failed to find stream info");
        return;
    }

    video_stream = av_find_best_stream(fmt_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    if (video_stream >= 0) {
        avcodec_parameters_copy(codecpar, fmt_ctx->streams[video_stream]->codecpar);
        time_base = fmt_ctx->streams[video_stream]->time_base;
    }
}

USBPacketProducer::~USBPacketProducer() {
    avformat_close_input(&fmt_ctx);
}

void USBPacketProducer::start() {
    TaskManager::start();
}

void USBPacketProducer::stop() {
    TaskManager::stop();
}

void USBPacketProducer::run() {
    while(running) {
        AVPacket pkt;
        while(running && av_read_frame(fmt_ctx, &pkt) >= 0) {
            if(pkt.stream_index == video_stream) {
                m_status.frame_cnt ++;
                m_status.timestamp = get_now_time();
                Packet pkt_copy1, pkt_copy2;
                pkt_copy1.m_data = av_packet_clone(&pkt);
                m_channel2rtsp->send(pkt_copy1); // 因为rtsp直接就向外部发送，所以可以不记录时间
                pkt_copy2.cam_id = cam_id;
                pkt_copy2.m_costTimes.when_get_packet[cam_id] = get_now_time();
                pkt_copy2.m_costTimes.image_frame_cnt[cam_id] = m_status.frame_cnt;
                pkt_copy2.m_data = av_packet_clone(&pkt);
                pkt_copy2.m_timestamp = get_now_time();
                m_channel2decoder->send(pkt_copy2);
            }
            av_packet_unref(&pkt);
        }
    }
}
