#pragma once
#include <string>
#include <vector>
#include <mutex>
#include <atomic>
#include <iostream>
#include <queue>
#include <chrono>

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
#include "image_decoder.h"
#include "image_encoder.h"
#include "AVFrameProducer.h"

// 是否可以看一下简单的工厂模式
class camera_manager {
public:
    static camera_manager* GetInstance();
    void start();
    void stop();
    safe_queue<Frame>& get_stitch_camera_stream(); // 相机拼接图
    safe_queue<Frame>& get_single_camera_sub_stream(int cam_id); // 单相机子码流，非拼接图
    safe_queue<Frame>& get_stitch_IR_camera_stream(); // 红外相机拼接图
private:
    camera_manager();
    ~camera_manager();
    std::vector<TaskManager*> m_task;
    StitchConsumer* stitch_handle;
    int cam_num{0};
};