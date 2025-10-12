# pragma once
#include "Consumer.h"
#include <vector>
extern "C" {
    #include "libavformat/avformat.h"
    #include "libavcodec/avcodec.h"
    #include "libavutil/pixfmt.h" 
    #include "libavutil/pixdesc.h" 
    #include "libavutil/opt.h"
    #include "libavutil/log.h"
    #include "libavcodec/bsf.h"
}

#include "safe_queue.hpp"
#include "tools.hpp"
#include "Stitch.h"
#include "config.h"
#include "image_encoder.h"
#include "safe_queue.hpp"
#include "LogConsumer.h"
#define FRAMECACHE_SIZE 10
struct FrameCache {
    std::atomic<int> m_cnt{0};
    Frame m_data;
    uint64_t m_timestamp[10]{0}; // 最终所有timestamp取平均送给m_data 
};
class StitchConsumer : public Consumer {
    std::vector<safe_queue<Frame>*> m_frame;
    std::vector<std::thread> m_threads; // 多路线程，分别做拼接
    FrameCache m_frameCache[FRAMECACHE_SIZE];
    safe_queue<Frame> frame_output;
    int cam_num{0};
    int width{0};
    int height{0};
    StitchStatus m_status{};
    std::string url;
    AVFormatContext* out_ctx;
    AVStream* out_stream;
    AVCodecParameters* codecpar;
    Stitch stitch;
    std::unique_ptr<TaskManager> m_rtspConsumer; // 拼接图像的推流线程，自己创建
    friend class LogConsumer;
    void single_stitch(int cam_id);
public:
    StitchConsumer(std::vector<safe_queue<Frame>*> frame_to_stitch, int width, int height);
    safe_queue<Frame>& get_stitch_frame();
    virtual ~StitchConsumer();
    virtual void start();
    virtual void stop();
    virtual void run();
};