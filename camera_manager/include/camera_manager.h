#pragma once
#include <string>
#include <vector>
#include <mutex>
#include <atomic>
#include <iostream>
#include <queue>
#include <chrono>
#include <acl/acl.h>

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
#include "rtsp.h"
#include "Stitch.h"
#include "image_decoder.h"
#include "image_encoder.h"

const int cam_num = (5);

class camera_manager {
public:
    static camera_manager* GetInstance();
// producer
    void rtspProducer(int cam_id);
    void videoFileProducer(int cam_id);
    void picFileProducer(int cam_id, std::string fileName, int height, int width, int fps); // 以图片作为stream的输入
// consumer
    void stitchConsumer();
    void rtspConsumer();
// utils
    void start();
    void join();
    void stop();
    safe_queue<std::pair<AVFrame*,costTimes>>& get_stitch_stream();
    void save_stream_to_file(int cam_id);
    void cout_message();
private:
    camera_manager();
    struct Camera_param {
        AVCodecParameters* codecpar; 
        AVRational time_base;
    };
    double camera_timestamp[cam_num] = {0.0};
    int camera_fps[cam_num] = {0};
    std::pair<int,int> camera_res[cam_num];
    struct Camera_param camera_para[cam_num];
    safe_queue<std::pair<AVPacket*,costTimes>> packet_input[cam_num];
    safe_queue<std::pair<AVFrame*,costTimes>> frame_input[cam_num];
    safe_queue<std::pair<AVFrame*,costTimes>> frame_output;
    safe_queue<std::pair<AVPacket*,costTimes>> packet_output;
    std::vector<std::thread> workers;
    std::atomic<bool> running{true}; // 全局运行标志
};
