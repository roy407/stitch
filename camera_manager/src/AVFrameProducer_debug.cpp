#include "AVFrameProducer_debug.h"
#include "RtspConsumer.h"

AVFrameProducer_debug::AVFrameProducer_debug(CameraConfig camera_config): AVFrameProducer() {
    this->cam_id = camera_config.cam_id;
    m_name += camera_config.name;
    fmt_ctx = avformat_alloc_context();
    av_dict_set(&options, "buffer_size", "4096000", 0);
    av_dict_set(&options, "rtsp_transport", "tcp", 0);
    av_dict_set(&options, "stimeout", "5000000", 0);
    // 从文件中获取数据
    cam_path = CFG_HANDLE.GetGlobalConfig().rtsp_record_path + std::to_string(cam_id) + ".mp4";
    m_status.width = camera_config.width;
    m_status.height = camera_config.height;

    {
        int ret = avformat_open_input(&fmt_ctx, cam_path.c_str(), nullptr, &options);
        if(ret < 0) return;
        if(avformat_find_stream_info(fmt_ctx, nullptr) >= 0) {
            video_stream = av_find_best_stream(fmt_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
            if(video_stream >= 0) {
                time_base = fmt_ctx->streams[video_stream]->time_base;
                codecpar = avcodec_parameters_alloc();
                avcodec_parameters_copy(codecpar, fmt_ctx->streams[video_stream]->codecpar);
            }
        }
        avformat_close_input(&fmt_ctx);
    }
    m_channel2rtsp = new PacketChannel;
    m_channel2decoder = new PacketChannel;
}

void AVFrameProducer_debug::run() {
    open_mp4:   // 循环打开的入口
    {
        int ret = avformat_open_input(&fmt_ctx, cam_path.c_str(), nullptr, &options);
        if(ret < 0) return;
        if(avformat_find_stream_info(fmt_ctx, nullptr) >= 0) {
            video_stream = av_find_best_stream(fmt_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
        }
    }

    auto start_time = std::chrono::steady_clock::now();
    double start_pts = AV_NOPTS_VALUE;

    AVPacket pkt;

    while (running)
    {
        int ret = av_read_frame(fmt_ctx, &pkt);
        if (ret < 0) {
            // === MP4 播放完毕，重新打开 ===
            avformat_close_input(&fmt_ctx);
            start_pts = AV_NOPTS_VALUE;
            goto open_mp4;  // 回到最开始重新读取
        }

        // 只处理视频帧
        if(pkt.stream_index == video_stream) {
            m_status.frame_cnt ++;
            m_status.timestamp = get_now_time();

            Packet pkt_copy1, pkt_copy2;
            pkt_copy2.cam_id = cam_id;
            pkt_copy2.m_costTimes.when_get_packet[cam_id] = get_now_time();
            pkt_copy2.m_costTimes.image_frame_cnt[cam_id] = m_status.frame_cnt;

            pkt_copy1.m_data = av_packet_clone(&pkt);
            pkt_copy2.m_data = av_packet_clone(&pkt);

            // ---- 保证 pts -> 实时播放（对齐原视频速度）----
            double pts_sec = pkt.pts * av_q2d(time_base);
            if (start_pts == AV_NOPTS_VALUE) {
                start_pts = pts_sec;
                start_time = std::chrono::steady_clock::now();
            }

            double relative_pts = pts_sec - start_pts;
            auto target_time = start_time + std::chrono::duration<double>(relative_pts);
            auto now = std::chrono::steady_clock::now();
            if (now < target_time) {
                std::this_thread::sleep_until(target_time);
            }

            pkt_copy2.m_timestamp = get_now_time();
            m_channel2rtsp->send(pkt_copy1);
            m_channel2decoder->send(pkt_copy2);
        }
        av_packet_unref(&pkt);
    }

    avformat_close_input(&fmt_ctx);
}
