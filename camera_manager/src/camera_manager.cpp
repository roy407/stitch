#include "camera_manager.h"
#include <vector>
#include <mutex>
#include <atomic>
#include <iostream>
#include <queue>
#include <sstream>

extern "C" {
    #include "libavformat/avformat.h"
    #include "libavcodec/avcodec.h"
    #include "libavutil/pixfmt.h" 
    #include "libavutil/pixdesc.h" 
    #include "libavutil/opt.h"
    #include "libavutil/log.h"
    #include "libavutil/error.h" 
    #include "libavcodec/bsf.h"
}
#include "safe_queue.hpp"
#include "image_decoder.h"
#include "image_encoder.h"
#include "config.h"

camera_manager* camera_manager::GetInstance() {
    static camera_manager cam;
    return &cam;
}

camera_manager::camera_manager() {
    cam_num = config::GetInstance().GetCameraConfig().size();
}

camera_manager::~camera_manager() {
    for(auto i: m_task) delete i;
}

void camera_manager::start() {
    avformat_network_init(); // 初始化网络模块
    std::vector<safe_queue<Frame>*> frames;
    LogConsumer* log = new LogConsumer();
    m_task.push_back(log);
    int width = 0;
    int height = 0;
    for(int i=0;i<cam_num;i++) {
        AVFrameProducer* pro = new AVFrameProducer(i);
        width = pro->getWidth();
        height = pro->getHeight();
        log->setProducer(pro);
        m_task.emplace_back(pro); // 对每个producer
        frames.push_back(&(pro->getFrameSender()));
    }
    stitch_handle = new StitchConsumer(frames, width, height);
    log->setConsumer(stitch_handle);
    m_task.emplace_back(stitch_handle);
    for(int i=0;i<m_task.size();i++) {
        m_task[i]->start();
    }
}

void camera_manager::stop() {
    for(int i=0;i<m_task.size();i++) {
        m_task[i]->stop();
    }
    avformat_network_deinit();
}

safe_queue<Frame> &camera_manager::get_stitch_IR_camera_stream() {
    
}

safe_queue<Frame> &camera_manager::get_stitch_camera_stream() {
    if(stitch_handle) {
        auto x = dynamic_cast<StitchConsumer*>(stitch_handle);
        return x->get_stitch_frame();
    } else {
        throw std::runtime_error("stitchConsumer is not init");
    }
}

safe_queue<Frame> &camera_manager::get_single_camera_sub_stream(int cam_id) {

}