#include "AVFrameProducer.h"

AVFrameProducer::AVFrameProducer() {
    
}

AVFrameProducer::AVFrameProducer(CameraConfig camera_config)
{
    this->cam_id = camera_config.cam_id;
    m_name += camera_config.name;
    fmt_ctx = avformat_alloc_context();
    av_dict_set(&options, "buffer_size", "4096000", 0);
    av_dict_set(&options, "rtsp_transport", "tcp", 0);
    av_dict_set(&options, "stimeout", "5000000", 0);
    cam_path = camera_config.input_url;
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

AVFrameProducer::AVFrameProducer(int cam_id, std::string name, std::string input_url, int width, int height) {
    this->cam_id = cam_id;
    m_name += name;
    fmt_ctx = avformat_alloc_context();
    av_dict_set(&options, "buffer_size", "4096000", 0);
    av_dict_set(&options, "rtsp_transport", "tcp", 0);
    av_dict_set(&options, "stimeout", "5000000", 0);
    cam_path = input_url;
    m_status.width = width;
    m_status.height = height;
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

AVFrameProducer::~AVFrameProducer() {
    avcodec_parameters_free(&codecpar);
    delete m_channel2rtsp;
    delete m_channel2decoder;
}

void AVFrameProducer::start() {
    TaskManager::start();
}
void AVFrameProducer::stop() {
    TaskManager::stop();
}

void AVFrameProducer::run() {
    while(running) {
        int ret = 0;
        ret = avformat_open_input(&fmt_ctx, cam_path.c_str(), nullptr, &options);
        if(ret < 0) {
            std::this_thread::sleep_for(std::chrono::seconds(3));
            continue;
        }
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
        avformat_close_input(&fmt_ctx);
    }
}

int AVFrameProducer::getWidth() const {
    return m_status.width;
}

int AVFrameProducer::getHeight() const {
    return m_status.height;
}

AVRational AVFrameProducer::getTimeBase() const {
    return time_base;
}

AVCodecParameters *AVFrameProducer::getAVCodecParameters() const {
    return codecpar;
}

PacketChannel *AVFrameProducer::getChannel2Rtsp() const {
    return m_channel2rtsp;
}

PacketChannel *AVFrameProducer::getChannel2Decoder() const {
    return m_channel2decoder;
}

