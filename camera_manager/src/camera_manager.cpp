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

camera_manager* camera_manager::GetInstance() {
    static camera_manager cam;
    return &cam;
}

camera_manager::camera_manager() {
    m_log = new LogConsumer();
    initPipeline();
}

camera_manager::~camera_manager() {
    delete m_log;
    for(auto& p : m_pipelines) delete p;
}

void camera_manager::start() {
    if(!m_running) {
        avformat_network_init(); // 初始化网络模块
        for(auto& p : m_pipelines) p->start();
        m_log->start();
        m_running = true;
    } else {
        LOG_WARN("camera manager already started");
    }
}

void camera_manager::stop() {
    if(m_running) {
        m_log->stop();
        for(auto& p : m_pipelines) p->stop();
        avformat_network_deinit();
        m_running = false;
    } else {
        LOG_WARN("camera manager already stopped");
    }
}

void camera_manager::initPipeline() {
    auto& cfg = CFG_HANDLE.GetConfig();
    Pipeline::setLogConsumer(m_log);
    for(auto& p : cfg.pipelines) {
        auto pipeline = new Pipeline(p);
        m_pipelines.emplace_back(pipeline);
    }
}

FrameChannel* camera_manager::getStitchCameraStream(int pipeline_id) {
    return m_pipelines[pipeline_id]->getStitchCameraStream();
}

FrameChannel* camera_manager::getSingleCameraSubStream(int cam_id) {
    for(auto& p : m_pipelines) {
        auto ptr = p->getResizeCameraStream(cam_id);
        if(ptr != nullptr) return ptr;
    }
    return nullptr;
}