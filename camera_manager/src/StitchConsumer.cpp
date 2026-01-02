#include "StitchConsumer.h"
#include "StitchImpl.h"

StitchConsumer::StitchConsumer(StitchOps* ops, int single_width, int height, int output_width) {
    m_name += "stitch";
    this->ops = ops;
    this->single_width = single_width;
    this->height = height;
    this->output_width = output_width;
    m_status.width = output_width;
    m_status.height = height;
    m_channel2show = new FrameChannel;
}

void StitchConsumer::setChannels(std::vector<FrameChannel*> channels) {
    m_channelsFromDecoder = channels;
}

FrameChannel *StitchConsumer::getChannel2Show() {
    return m_channel2show;
}

StitchConsumer::~StitchConsumer() {
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
    AVFrame** inputs = new AVFrame*[MAX_CAM_SIZE];
    LOG_DEBUG("total count {} channels", m_channelsFromDecoder.size());
    #if defined(KERNEL_TEST)
    int frame_index = 0;
    uint64_t frame_cnt = 0; // 用于模拟计数每帧图像
    std::vector<int> cam_ids;
    for (auto& channel : m_channelsFromDecoder) {
        Frame tmp;
        if(!channel->recv(tmp)) goto cleanup;
        inputs[frame_index] = tmp.m_data;
        cam_ids.push_back(tmp.cam_id);
        frame_index ++;
    }
    #endif
    while (running) {
        #if !defined(KERNEL_TEST)
        int frame_index = 0;
        for (auto& channel : m_channelsFromDecoder) {
            Frame tmp;
            if(!channel->recv(tmp)) goto cleanup;
            inputs[frame_index] = tmp.m_data;
            out_image.m_costTimes.image_frame_cnt[tmp.cam_id] = tmp.m_costTimes.image_frame_cnt[tmp.cam_id];
            out_image.m_costTimes.when_get_packet[tmp.cam_id] = tmp.m_costTimes.when_get_packet[tmp.cam_id];
            out_image.m_costTimes.when_get_decoded_frame[tmp.cam_id] = tmp.m_costTimes.when_get_decoded_frame[tmp.cam_id];
            frame_index ++;
        }
        #else
        for (int i=0; i < frame_index; i++) {
            out_image.m_costTimes.image_frame_cnt[cam_ids[i]] = frame_cnt ++;
            out_image.m_costTimes.when_get_packet[cam_ids[i]] = get_now_time(); // 模拟获取packet过程
            out_image.m_costTimes.when_get_decoded_frame[cam_ids[i]] = out_image.m_costTimes.when_get_packet[cam_ids[i]]; // 模拟packet解码过程
        }
        #endif
        out_image.m_data = ops->stitch(ops->obj, inputs);
        out_image.m_data->pts = inputs[0]->pts;
        out_image.m_costTimes.when_get_stitched_frame = get_now_time();
        m_channel2show->send(out_image);
        m_status.frame_cnt ++;
        m_status.timestamp = get_now_time();
        #if !defined(KERNEL_TEST)
        for (int i = 0; i < m_channelsFromDecoder.size(); ++i) {
            if (inputs[i]) {
                av_frame_free(&inputs[i]);
            }
        }
        #endif
    }
cleanup:
    for(auto& channel : m_channelsFromDecoder) {
        channel->clear();
        #if defined(KERNEL_TEST)
        delete channel;
        #endif
    }
    delete[] inputs;
}
