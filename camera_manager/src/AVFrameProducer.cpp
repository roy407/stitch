#include "AVFrameProducer.h"
#include "RtspConsumer.h"

AVFrameProducer::AVFrameProducer(int cam_id) {
    this->cam_id = cam_id;
    m_name += std::to_string(this->cam_id);
    fmt_ctx = avformat_alloc_context();
    av_dict_set(&options, "rtsp_transport", "udp", 0);
    av_dict_set(&options, "stimeout", "5000000", 0);
    const std::string status = config::GetInstance().GetGlobalConfig().status;
    if(status == "rtsp") { // 支持RTSP取流
        if(config::GetInstance().GetGlobalConfig().use_sub_input) {
            cam_path = config::GetInstance().GetCameraConfig()[cam_id].sub_input_url;
        } else {
            cam_path = config::GetInstance().GetCameraConfig()[cam_id].input_url;
        }
    } else if(status == "file") { // 文件读取暂不支持
        cam_path = config::GetInstance().GetGlobalConfig().save_rtsp_data_path + std::to_string(cam_id) + ".mp4";
    }
    int ret = 0;
    do {
        ret = avformat_open_input(&fmt_ctx, cam_path.c_str(), nullptr, &options);
        if(ret < 0) {
            std::this_thread::sleep_for(std::chrono::seconds(3));
            continue;
        }
    } while(ret);
    if(avformat_find_stream_info(fmt_ctx, nullptr) >= 0) {
        video_stream = av_find_best_stream(fmt_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
        if(video_stream >= 0) {
            stream = fmt_ctx->streams[video_stream];
            codecpar = stream->codecpar;
        } else {
            throw std::runtime_error("error");
        }
    } else {
        throw std::runtime_error("error");
    }

    m_status.width = codecpar->width;
    m_status.height = codecpar->height;

    if(config::GetInstance().GetCameraConfig()[cam_id].rtsp) {
        m_rtspConsumer = std::make_unique<RtspConsumer>(m_packetSender1, &codecpar, &(stream->time_base), config::GetInstance().GetCameraConfig()[cam_id].output_url);
    } else {
        m_rtspConsumer = std::make_unique<Consumer>(); // 如果不推rtsp，那么就创建空的Consumer
    }
}

AVFrameProducer::~AVFrameProducer() {
    avformat_close_input(&fmt_ctx);
}

void AVFrameProducer::start() {
    img_dec.start_image_decoder(codecpar, &m_frameSender, &m_packetSender2);
    TaskManager::start();
    m_rtspConsumer->start();
}
void AVFrameProducer::stop() {
    m_rtspConsumer->stop();
    TaskManager::stop();
}

void AVFrameProducer::run() {
    while(running) {
        AVPacket pkt;
        while(running && av_read_frame(fmt_ctx, &pkt) >= 0) {
            if(pkt.stream_index == video_stream) {
                Packet pkt_copy1, pkt_copy2;
                pkt_copy1.m_data = av_packet_clone(&pkt);
                pkt_copy2.m_data = av_packet_clone(&pkt);
                m_packetSender1.push(pkt_copy1);
                m_packetSender2.push(pkt_copy2);
                m_status.frame_cnt ++;
            }
            av_packet_unref(&pkt);
        }
        avformat_close_input(&fmt_ctx);

        int ret = 0;
        ret = avformat_open_input(&fmt_ctx, cam_path.c_str(), nullptr, &options);
        if(ret < 0) {
            std::this_thread::sleep_for(std::chrono::seconds(3));
            continue;
        }
        if(avformat_find_stream_info(fmt_ctx, nullptr) >= 0) {
            video_stream = av_find_best_stream(fmt_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
            if(video_stream >= 0) {
                stream = fmt_ctx->streams[video_stream];
                codecpar = stream->codecpar;
            }
        }
    }
}

int AVFrameProducer::getWidth() const {
    return m_status.width;
}

int AVFrameProducer::getHeight() const {
    return m_status.height;
}

safe_queue<Frame> &AVFrameProducer::getFrameSender() {
    return m_frameSender;
}

safe_queue<Packet> &AVFrameProducer::getPacketSender() {
    return m_packetSender1;
}
