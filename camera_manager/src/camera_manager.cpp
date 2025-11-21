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
#include "StitchImpl.h"

camera_manager* camera_manager::GetInstance() {
    static camera_manager cam;
    return &cam;
}

camera_manager::camera_manager() {
    camera_num = config::GetInstance().GetCameraConfig().size();
    IR_camera_num = config::GetInstance().GetIRCameraConfig().size();
    log = new LogConsumer();
    create_channel_1();
    create_channel_2();
    create_channel_3();
}

camera_manager::~camera_manager() {
    delete log;
    for(auto i: m_consumer_task) delete i;
    for(auto i: m_producer_task) delete i;
    destory_channel_3();
    destory_channel_2();
    destory_channel_1();
}

void camera_manager::start() {
    avformat_network_init(); // 初始化网络模块
    for(int i=0;i<m_producer_task.size();i++) {
        m_producer_task[i]->start();
    }
    for(int i=0;i<m_consumer_task.size();i++) {
        m_consumer_task[i]->start();
    }
    log->start();
}

void camera_manager::stop() {
    log->stop();
    for(int i=0;i<m_producer_task.size();i++) {
        m_producer_task[i]->stop();
    }
    for(int i=0;i<m_consumer_task.size();i++) {
        m_consumer_task[i]->stop();
    }
    avformat_network_deinit();
}

void camera_manager::create_channel_1() {
    std::vector<safe_queue<Frame>*> frames;
    const std::vector<CameraConfig> cameras = config::GetInstance().GetCameraConfig();
    int width = 0; // 默认都是大小相同的相机
    int height = 0;
    for(int i = 0;i < camera_num;i ++) {
        AVFrameProducer* pro = new AVFrameProducer(cameras[i]);
        width = pro->getWidth();
        height = pro->getHeight();
        m_producer_task.emplace_back(pro);
        log->setProducer(pro);
        frames.push_back(&(pro->getFrameSender()));
    }
    int output_width = config::GetInstance().GetGlobalStitchConfig().camera_stitch_output_width;
    StitchOps* ops = make_stitch_ops(new StitchImpl<YUV420, MappingTableKernel>());
    ops->init(ops->obj, frames.size(), width, output_width, height);
    StitchConsumer * stitch = new StitchConsumer(ops, frames, width, height, output_width);
    channel_1_output = stitch;
    log->setConsumer(stitch);
    opses.push_back(ops);
    m_consumer_task.push_back(stitch);
}

void camera_manager::destory_channel_1() {

}

void camera_manager::create_channel_2() {
    std::vector<safe_queue<Frame>*> frames;
    const std::vector<IRCameraConfig> IR_cameras = config::GetInstance().GetIRCameraConfig();
    int width = 0; // 默认都是大小相同的相机
    int height = 0;
    for(int i = 0;i < IR_camera_num;i ++) {
        AVFrameProducer* pro = new AVFrameProducer(IR_cameras[i]);
        width = pro->getWidth();
        height = pro->getHeight();
        m_producer_task.emplace_back(pro);
        frames.push_back(&(pro->getFrameSender()));
    }
    int output_width = config::GetInstance().GetGlobalStitchConfig().IR_camera_stitch_output_width;
    StitchOps* ops = make_stitch_ops(new StitchImpl<YUV420, RawKernel>());
    ops->init(ops->obj, frames.size(), width, output_width, height);
    StitchConsumer * stitch = new StitchConsumer(ops, frames, width, height, output_width);
    channel_2_output = stitch;
    opses.push_back(ops);
    m_consumer_task.push_back(stitch);
}

void camera_manager::destory_channel_2() {
    delete_stitch_ops<StitchImpl<YUV420, MappingTableKernel>>(opses[0]);
}

void camera_manager::create_channel_3() {
    const std::vector<CameraConfig> cameras = config::GetInstance().GetCameraConfig();
    for(int i = 0;i < camera_num;i ++) {
        AVFrameProducer* pro = new AVFrameProducer(cameras[i].cam_id, cameras[i].name,cameras[i].sub.input_url, cameras[i].sub.width, cameras[i].sub.height);
        m_producer_task.emplace_back(pro);
        auto& x = pro->getFrameSender();
        m_sub_stream.push_back(&x);
    }
}

void camera_manager::destory_channel_3() {
    delete_stitch_ops<StitchImpl<YUV420, RawKernel>>(opses[1]);
}

safe_queue<Frame> &camera_manager::get_stitch_IR_camera_stream() {
    auto x = dynamic_cast<StitchConsumer*>(channel_2_output);
    return x->get_stitch_frame();
}

safe_queue<Frame> &camera_manager::get_stitch_camera_stream() {
    auto x = dynamic_cast<StitchConsumer*>(channel_1_output);
    return x->get_stitch_frame();
}

safe_queue<Frame> &camera_manager::get_single_camera_sub_stream(int cam_id) {
    return *m_sub_stream[cam_id];
}