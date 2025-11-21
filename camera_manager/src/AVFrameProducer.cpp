#include "AVFrameProducer.h"
#include "RtspConsumer.h"

void AVFrameProducer::setDecoder(std::string decoder_name) {
    img_dec = new image_decoder;
}

AVFrameProducer::AVFrameProducer(CameraConfig camera_config) {
    this->cam_id = camera_config.cam_id;
    m_name += camera_config.name;
    fmt_ctx = avformat_alloc_context();
    av_dict_set(&options, "buffer_size", "4096000", 0);
    av_dict_set(&options, "rtsp_transport", "tcp", 0);
    av_dict_set(&options, "stimeout", "5000000", 0);
    cam_path = camera_config.input_url;
    m_status.width = camera_config.width;
    m_status.height = camera_config.height;
    rtsp = camera_config.rtsp;
    setDecoder("");
    {
        int ret = avformat_open_input(&fmt_ctx, cam_path.c_str(), nullptr, &options);
        if(ret < 0) return;
        if(avformat_find_stream_info(fmt_ctx, nullptr) >= 0) {
            video_stream = av_find_best_stream(fmt_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
            if(video_stream >= 0) {
                stream = fmt_ctx->streams[video_stream];
                codecpar = stream->codecpar;
                img_dec->start_image_decoder(cam_id, codecpar, &m_frameSender, &m_packetSender2);
                if(rtsp) {
                    m_rtspConsumer = std::make_unique<RtspConsumer>(m_packetSender1, &codecpar, &(stream->time_base), config::GetInstance().GetCameraConfig()[cam_id].output_url);
                    m_rtspConsumer->start();
                }
            }
        }
        avformat_close_input(&fmt_ctx);
    }
}

AVFrameProducer::AVFrameProducer(IRCameraConfig IR_camera_config) {
    this->cam_id = IR_camera_config.cam_id;
    m_name += IR_camera_config.name;
    fmt_ctx = avformat_alloc_context();
    av_dict_set(&options, "buffer_size", "4096000", 0);
    av_dict_set(&options, "rtsp_transport", "tcp", 0);
    av_dict_set(&options, "stimeout", "5000000", 0);
    cam_path = IR_camera_config.input_url;
    m_status.width = IR_camera_config.width;
    m_status.height = IR_camera_config.height;
    rtsp = IR_camera_config.rtsp;
    setDecoder("");
    {
        int ret = avformat_open_input(&fmt_ctx, cam_path.c_str(), nullptr, &options);
        if(ret < 0) return;
        if(avformat_find_stream_info(fmt_ctx, nullptr) >= 0) {
            video_stream = av_find_best_stream(fmt_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
            if(video_stream >= 0) {
                stream = fmt_ctx->streams[video_stream];
                codecpar = stream->codecpar;
                img_dec->start_image_decoder(cam_id, codecpar, &m_frameSender, &m_packetSender2);
                if(rtsp) {
                    m_rtspConsumer = std::make_unique<RtspConsumer>(m_packetSender1, &codecpar, &(stream->time_base), config::GetInstance().GetCameraConfig()[cam_id].output_url);
                    m_rtspConsumer->start();
                }
            }
        }
        avformat_close_input(&fmt_ctx);
    }
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
    setDecoder("");
    {
        int ret = avformat_open_input(&fmt_ctx, cam_path.c_str(), nullptr, &options);
        if(ret < 0) return;
        if(avformat_find_stream_info(fmt_ctx, nullptr) >= 0) {
            video_stream = av_find_best_stream(fmt_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
            if(video_stream >= 0) {
                stream = fmt_ctx->streams[video_stream];
                codecpar = stream->codecpar;
                img_dec->start_image_decoder(cam_id, codecpar, &m_frameSender, &m_packetSender2);
                if(rtsp) {
                    m_rtspConsumer = std::make_unique<RtspConsumer>(m_packetSender1, &codecpar, &(stream->time_base), config::GetInstance().GetCameraConfig()[cam_id].output_url);
                    m_rtspConsumer->start();
                }
            }
        }
        avformat_close_input(&fmt_ctx);
    }
}

AVFrameProducer::~AVFrameProducer() {
}

void AVFrameProducer::start() {
    TaskManager::start();
}
void AVFrameProducer::stop() {
    if(rtsp) {
        m_rtspConsumer->stop();
    }
    TaskManager::stop();
    img_dec->close_image_decoder();
    delete img_dec;
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
                pkt_copy2.cam_id = cam_id;
                pkt_copy2.m_costTimes.when_get_packet[cam_id] = get_now_time();
                pkt_copy2.m_costTimes.image_frame_cnt[cam_id] = m_status.frame_cnt;
                pkt_copy1.m_data = av_packet_clone(&pkt);
                pkt_copy2.m_data = av_packet_clone(&pkt);
                pkt_copy2.m_timestamp = get_now_time(); // 获取当前时间戳
                m_packetSender1.push(pkt_copy1);
                m_packetSender2.push(pkt_copy2);
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

safe_queue<Frame> &AVFrameProducer::getFrameSender() {
    return m_frameSender;
}

safe_queue<Packet> &AVFrameProducer::getPacketSender() {
    return m_packetSender1;
}
