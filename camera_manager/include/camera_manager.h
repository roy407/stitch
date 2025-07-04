#include <string>
#include <vector>
#include <mutex>
#include <atomic>
#include <iostream>
#include <queue>
#include <chrono>
#include <cuda_runtime.h>

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
#include "rtsp.h"
#include "Stitch.h"
#include "image_decoder.h"
#include "image_encoder.h"

#define cam_num (2)

class camera_manager {
public:
    camera_manager();
    void get_stream_from_rtsp(int cam_id);
    void get_stream_from_file(int cam_id);
    void save_stream_to_file(int cam_id);
    void do_stitch();
    void start();
    void stop();
    safe_queue<AVFrame*>& get_stitch_stream();
    void cout_message();
private:
    struct Camera_param {
        AVCodecParameters* codecpar; 
        AVRational time_base;
    };
    double camera_timestamp[cam_num] = {0.0};
    int camera_fps[cam_num] = {0};
    std::pair<int,int> camera_res[cam_num];
    struct Camera_param camera_para[cam_num];
    safe_queue<AVPacket*> packet_input[cam_num];
    safe_queue<AVFrame*> frame_input[cam_num];
    safe_queue<AVFrame*> frame_output;
    safe_queue<AVPacket*> packet_output;
    std::atomic<bool> running{true}; // 全局运行标志
};