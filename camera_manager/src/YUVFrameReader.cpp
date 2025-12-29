// YUVFrameReader.cpp
#include "YUVFrameReader.h"
extern "C" {
    #include <libavutil/imgutils.h>
}

YUVFrameReader::YUVFrameReader(int cam_id, const std::string& yuv_path, int w, int h) 
    : cam_id(cam_id), yuv_file_path(yuv_path), width(w), height(h) {
    m_name = "YUVFrameReader_" + std::to_string(cam_id);
    m_channel2stitch = new FrameChannel;
    m_channel2resize = new FrameChannel;
}

YUVFrameReader::~YUVFrameReader() {
    delete m_channel2stitch;
    delete m_channel2resize;
}

void YUVFrameReader::start() {
    file_stream.open(yuv_file_path, std::ios::binary);
    if (!file_stream.is_open()) {
        LOG_ERROR("Cannot open YUV file: {}", yuv_file_path);
        return;
    }
    TaskManager::start();
}

void YUVFrameReader::stop() {
    TaskManager::stop();
    if (file_stream.is_open()) {
        file_stream.close();
    }
}

void YUVFrameReader::run() {
    // 计算一帧YUV420P的大小
    int y_size = width * height;
    int uv_size = y_size / 4;
    int frame_size = y_size + uv_size * 2;
    
    std::vector<uint8_t> buffer(frame_size);
    auto frame_interval = std::chrono::duration<double>(1.0 / fps);
    
    while (running && file_stream) {
        auto start_time = std::chrono::steady_clock::now();
        
        // 读取一帧YUV数据
        file_stream.read(reinterpret_cast<char*>(buffer.data()), frame_size);
        if (file_stream.gcount() != frame_size) {
            // 文件结束，重新开始
            file_stream.clear();
            file_stream.seekg(0, std::ios::beg);
            continue;
        }
        
        // 创建AVFrame并填充数据
        Frame frame;
        frame.cam_id = cam_id;
        frame.m_data = av_frame_alloc();
        frame.m_data->width = width;
        frame.m_data->height = height;
        frame.m_data->format = AV_PIX_FMT_YUV420P;
        
        // 分配帧内存
        if (av_frame_get_buffer(frame.m_data, 32) < 0) {
            LOG_ERROR("Cannot allocate frame buffer");
            av_frame_free(&frame.m_data);
            continue;
        }
        
        // 复制YUV数据
        memcpy(frame.m_data->data[0], buffer.data(), y_size);
        memcpy(frame.m_data->data[1], buffer.data() + y_size, uv_size);
        memcpy(frame.m_data->data[2], buffer.data() + y_size + uv_size, uv_size);
        
        // 设置时间信息
        frame.m_timestamp = get_now_time();
        frame.m_costTimes.image_frame_cnt[cam_id] = ++frame_count;
        frame.m_costTimes.when_get_decoded_frame[cam_id] = get_now_time();
        
        // 发送到两个channel
        m_channel2stitch->send(frame);
        
        Frame frame_copy;
        frame_copy.cam_id = frame.cam_id;
        frame_copy.m_costTimes = frame.m_costTimes;
        frame_copy.m_timestamp = frame.m_timestamp;
        frame_copy.m_data = av_frame_clone(frame.m_data);
        m_channel2resize->send(frame_copy);
        
        // LOG_DEBUG("不控制帧率");
        // 控制帧率
        auto elapsed = std::chrono::steady_clock::now() - start_time;
        if (elapsed < frame_interval) {
            std::this_thread::sleep_for(frame_interval - elapsed);
        }
    }
}