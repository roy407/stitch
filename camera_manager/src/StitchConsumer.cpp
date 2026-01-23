#include "StitchConsumer.h"
#include "StitchImpl.h"
#include "log.hpp"

StitchConsumer::StitchConsumer(StitchOps* ops, int single_width, int height, int output_width) {
    m_name += "stitch";
    this->ops = ops;
    this->single_width = single_width;
    this->height = height;
    this->output_width = output_width;
    
    m_status.width = output_width;
    m_status.height = height;
    m_channel2show = new FrameChannel;
    
    // 初始化共享内存（发送端）
    shm_buffer_ = std::make_unique<StitchCircularBuffer>();
    std::string shm_name = "stitch_pipeline";
    shm_buffer_->initialize(shm_name, output_width, height, true);  // create_new=true
    if (shm_buffer_->is_ready()) {
        LOG_INFO("StitchConsumer SHM initialized: {} ({}x{})", shm_name, output_width, height);
    } else {
        LOG_ERROR("StitchConsumer SHM initialization failed: {}", shm_name);
    }
}

void StitchConsumer::setChannels(std::vector<FrameChannel*> channels) {
    m_channelsFromDecoder = channels;
}

FrameChannel *StitchConsumer::getChannel2Show() {
    return m_channel2show;
}

StitchConsumer::~StitchConsumer() {
    if (shm_buffer_) {
        shm_buffer_->cleanup();
    }
    delete m_channel2show;
}

void StitchConsumer::start() {
    TaskManager::start();
}

void StitchConsumer::stop() {
    for(auto& i: m_channelsFromDecoder) {
        i->stop();
    }
    TaskManager::stop();
}

void StitchConsumer::run() { 
    Frame out_image;
    AVFrame** inputs = new AVFrame*[10];
    LOG_DEBUG("total count {} channels", m_channelsFromDecoder.size());
    while (running) {
        int frame_size = 0;
        for (auto& channel : m_channelsFromDecoder) {
            Frame tmp;
            if(!channel->recv(tmp)) goto cleanup;
            inputs[frame_size] = tmp.m_data;
            out_image.m_costTimes.image_frame_cnt[tmp.cam_id] = tmp.m_costTimes.image_frame_cnt[tmp.cam_id];
            out_image.m_costTimes.when_get_packet[tmp.cam_id] = tmp.m_costTimes.when_get_packet[tmp.cam_id];
            out_image.m_costTimes.when_get_decoded_frame[tmp.cam_id] = tmp.m_costTimes.when_get_decoded_frame[tmp.cam_id];
            frame_size ++;
        }
        out_image.m_data = ops->stitch(ops->obj, inputs);
        out_image.m_data->pts = inputs[0]->pts;
        out_image.m_costTimes.when_get_stitched_frame = get_now_time();
        m_channel2show->send(out_image);
        
        // 推送到共享内存
        if (shm_buffer_ && shm_buffer_->is_ready()) {
            shm_buffer_->push_stitch_image(out_image.m_data);
        }
        
        m_status.frame_cnt ++;
        m_status.timestamp = get_now_time();
        for (int i = 0; i < m_channelsFromDecoder.size(); ++i) {
            if (inputs[i]) {
                av_frame_free(&inputs[i]);
            }
        }
    }
cleanup:
    for(auto& channel : m_channelsFromDecoder) {
        channel->clear();
    }
    delete[] inputs;
}
