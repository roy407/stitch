#pragma once
#include "PacketProducer.h"
#include "config.h"
#include <atomic>
#include <thread>
#include <vector>
#include <string>
#include <chrono>

extern "C" {
    #include "libavformat/avformat.h"
    #include "libavcodec/avcodec.h"
    #include "libavutil/pixfmt.h"
}

class TestPacketProducer : public PacketProducer {
public:
    TestPacketProducer(const CameraConfig& camera_config);
    virtual ~TestPacketProducer();
    
    void start() override;
    void stop() override;
    
    // 测试专用方法
    void setFrameRate(int fps);
    void setTestPattern(int pattern);
    
protected:
    void run() override;
    
private:
    void generateTestPacket(AVPacket& pkt);
    void updateStatus();  // 更新状态信息
    
    std::string m_name;
    int m_width = 1920;
    int m_height = 1080;
    
    // 测试参数
    int m_frameRate = 30;
    int m_pattern = 0;
    uint64_t m_frameCount = 0;
    
    // 测试数据缓冲区
    std::vector<uint8_t> m_testData;
    int m_testDataSize = 10000;
    
    // 线程控制
    std::atomic<bool> running{false};
    std::thread m_thread;
    
    // 时间控制
    std::chrono::steady_clock::time_point m_lastFrameTime;
    std::chrono::steady_clock::time_point m_startTime;
};


