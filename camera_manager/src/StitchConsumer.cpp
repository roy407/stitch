#include "StitchConsumer.h"
#include "StitchImpl.h"

StitchConsumer::StitchConsumer(StitchOps* ops, int single_width, int height, int output_width, std::string shm_name) {
    m_name += "stitch";
    this->ops = ops;
    this->single_width = single_width;
    this->height = height;
    this->output_width = output_width;
    m_status.width = output_width;
    m_status.height = height;
    m_channel2show = new FrameChannel();
    m_channel2rtsp = new FrameChannel();

    if (!shm_name.empty()) {
        m_shm_sender = new ShmSender(shm_name);
        if (!m_shm_sender->init()) {
            LOG_ERROR("Failed to init Shared Memory Sender!");
            // 初始化失败时，应该清理并设置为空，避免非法访问
            // 但init失败时m_shm_sender对象仍有效，只是sendFrame会检查m_shm_fd
            // 建议：如果不需要抛出异常，这里只打日志即可
        }
    }
}

void StitchConsumer::setChannels(std::vector<FrameChannel*> channels) {
    m_channelsFromDecoder = channels;
}

FrameChannel* StitchConsumer::getChannel2Show() {
    return m_channel2show;
}

FrameChannel* StitchConsumer::getChannel2Rtsp() {
    return m_channel2rtsp;
}

StitchConsumer::~StitchConsumer() {
    if(m_channel2show) delete m_channel2show;
    if(m_channel2rtsp) delete m_channel2rtsp;
    if(m_shm_sender) delete m_shm_sender;
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
    uint32_t fetched_cnt_when_stop = 0;
    #if defined(KERNEL_TEST)
    int frame_index = 0;
    uint64_t frame_cnt = 0; // 用于模拟计数每帧图像
    std::vector<int> cam_ids;
    for (auto& channel : m_channelsFromDecoder) {
        Frame tmp;
        if(!channel->recv(tmp)) goto cleanup;
        fetched_cnt_when_stop ++;    // 当运行了goto cleanup时，记录已经获取的帧数，这些帧数需要释放，否则会造成内存泄漏
        inputs[frame_index] = tmp.m_data;
        cam_ids.push_back(tmp.cam_id);
        frame_index ++;
    }
    #endif
    while (running) {
        #if !defined(KERNEL_TEST)
        int frame_index = 0;
        fetched_cnt_when_stop = 0;
        for (auto& channel : m_channelsFromDecoder) {
            Frame tmp;
            if(!channel->recv(tmp)) goto cleanup;
            fetched_cnt_when_stop ++;    // 当运行了goto cleanup时，记录已经获取的帧数，这些帧数需要释放，否则会造成内存泄漏
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
        
        if (out_image.m_data) {
            if (m_shm_sender) {
                m_shm_sender->sendFrame(out_image.m_data);
            }
            if (m_channel2show) {
                Frame refFrame = out_image;
                refFrame.m_data = av_frame_alloc();
                if (refFrame.m_data) {
                    av_frame_ref(refFrame.m_data, out_image.m_data);
                    m_channel2show->send(refFrame);
                }
            }
            if (m_channel2rtsp) {
                Frame refFrame = out_image;
                refFrame.m_data = av_frame_alloc();
                if (refFrame.m_data) {
                    av_frame_ref(refFrame.m_data, out_image.m_data);
                    m_channel2rtsp->send(refFrame);
                }
            }
            av_frame_unref(out_image.m_data);
            av_frame_free(&out_image.m_data); 
        }
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
    for(int i = 0; i < fetched_cnt_when_stop; ++i) {
        if (inputs[i]) {
            av_frame_free(&inputs[i]);
        }
    }
    for(auto& channel : m_channelsFromDecoder) {
        channel->clear();
        #if defined(KERNEL_TEST)
        delete channel;
        #endif
    }
    delete[] inputs;
}
