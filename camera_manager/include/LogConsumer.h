#pragma once

#include "Consumer.h"
#include <vector>
#include <mutex>

// 每一帧图像的具体信息
struct CamStatus {
    int width;
    int height;
    uint64_t timestamp;
    uint64_t frame_cnt;
};

struct StitchStatus {
    int width;
    int height;
    uint64_t timestamp;
    uint64_t frame_cnt;
};

class AVFrameProducer;
class StitchConsumer;

class LogConsumer : public Consumer {
    uint64_t m_time{0};
    int time_gap {2}; // 由此控制log的打印速度
    // 这个类对下面两个成员变量指向的两个类，只有观测权，没有修改权
    std::vector<AVFrameProducer*> m_pro;
    std::vector<StitchConsumer*> m_con;
    void printProducer(AVFrameProducer* pro, uint64_t& prev_frame_cnt, uint64_t& prev_timestamp);
    void printConsumer(StitchConsumer* con, uint64_t& prev_frame_cnt, uint64_t& prev_timestamp);
    void printGPUStatus();
    void printCPUStatus();
public:
    LogConsumer();
    ~LogConsumer();
    virtual void start();
    virtual void stop();
    virtual void run();
    void setProducer(AVFrameProducer* pro);
    void setConsumer(StitchConsumer* con);
};