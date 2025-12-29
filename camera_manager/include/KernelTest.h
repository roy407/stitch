#pragma once
#include "config.h"
#include "PacketProducer.h"

class KernelProducer : public PacketProducer {
protected:
    AVFormatContext* fmt_ctx{nullptr};
    AVDictionary* options{nullptr};
    std::string cam_path;
    int video_stream{-1};
    // 测试数据缓冲区
    std::vector<uint8_t> m_testData;
    int m_testDataSize = 0;
public:
    KernelProducer(CameraConfig camera_config);
    virtual ~KernelProducer();
    virtual void start();
    virtual void stop();
    virtual void run();
};