#include "TestPacketProducer.h"
#include "tools.hpp"
#include <chrono>
#include <thread>
#include <cstring>
#include <cstdlib>

TestPacketProducer::TestPacketProducer(const CameraConfig& camera_config) {
    this->cam_id = camera_config.cam_id;
    m_name = "TestPacketProducer_" + std::to_string(cam_id);
    
    // 设置状态
    m_status.width = camera_config.width;
    m_status.height = camera_config.height;
    m_status.frame_cnt = 0;
    m_status.timestamp = 0;
    
    m_width = camera_config.width;
    m_height = camera_config.height;
    
    // 创建通道
    m_channel2rtsp = new PacketChannel();
    m_channel2decoder = new PacketChannel();
    
    // 初始化编码参数（模拟H.264）
    codecpar = avcodec_parameters_alloc();
    if (codecpar) {
        codecpar->codec_type = AVMEDIA_TYPE_VIDEO;
        codecpar->codec_id = AV_CODEC_ID_H264;
        codecpar->width = m_width;
        codecpar->height = m_height;
        codecpar->format = AV_PIX_FMT_YUV420P;
    }
    
    // 初始化时间基准
    time_base = {1, 90000};  // 90kHz
    
    // 初始化测试数据
    m_testDataSize = 10000;
    m_testData.resize(m_testDataSize);
    
    // 填充测试数据
    for (int i = 0; i < m_testDataSize; i++) {
        m_testData[i] = i % 256;
    }
    
    m_startTime = std::chrono::steady_clock::now();
    m_lastFrameTime = m_startTime;
}

TestPacketProducer::~TestPacketProducer() {
    stop();
    if (codecpar) {
        avcodec_parameters_free(&codecpar);
    }
    if (m_channel2decoder) {
        delete m_channel2decoder;
    }
    if (m_channel2rtsp) {
        delete m_channel2rtsp;
    }
}

void TestPacketProducer::start() {
    if (!running) {
        running = true;
        m_thread = std::thread(&TestPacketProducer::run, this);
    }
}

void TestPacketProducer::stop() {
    running = false;
    if (m_thread.joinable()) {
        m_thread.join();
    }
}

void TestPacketProducer::updateStatus() {
    // 更新帧计数
    m_status.frame_cnt = m_frameCount;
    
    // 获取当前时间（纳秒）
    auto now = std::chrono::steady_clock::now();
    auto duration = now.time_since_epoch();
    m_status.timestamp = std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();
}

void TestPacketProducer::run() {
    while (running) {
        // 创建测试包
        AVPacket pkt;
        av_init_packet(&pkt);
        
        // 分配数据
        static const int kPacketSize = 1024;
        pkt.data = (uint8_t*)av_malloc(kPacketSize);
        if (!pkt.data) {
            LOG_ERROR("Failed to allocate packet data");
            break;
        }
        
        // 快速填充测试数据
        for (int i = 0; i < kPacketSize; i++) {
            pkt.data[i] = (m_frameCount + i) % 256;
        }
        pkt.size = kPacketSize;
        
        // 更新时间戳
        m_frameCount++;
        pkt.pts = m_frameCount;
        pkt.dts = m_frameCount;
        pkt.duration = 1;
        
        // 创建包的副本
        Packet pkt_copy1, pkt_copy2;
        pkt_copy2.cam_id = cam_id;
        
        uint64_t now_time = get_now_time();
        pkt_copy2.m_costTimes.when_get_packet[cam_id] = now_time;
        pkt_copy2.m_costTimes.image_frame_cnt[cam_id] = m_frameCount;
        
        // 克隆包
        pkt_copy1.m_data = av_packet_clone(&pkt);
        pkt_copy2.m_data = av_packet_clone(&pkt);
        
        // 更新状态
        updateStatus();
        
        // 直接发送（如果队列满可能会阻塞）
        // 为了最大性能，我们先检查是否有空间
        if (m_channel2rtsp && m_channel2decoder) {
            // 直接发送，让队列机制处理
            m_channel2rtsp->send(pkt_copy1);
            m_channel2decoder->send(pkt_copy2);
            
        }
        
        // 清理
        av_packet_unref(&pkt);
    }
}

void TestPacketProducer::setFrameRate(int fps) {
    m_frameRate = fps;
    LOG_INFO("TestPacketProducer {}: Frame rate set to {} fps", cam_id, fps);
}

void TestPacketProducer::setTestPattern(int pattern) {
    m_pattern = pattern;
}


