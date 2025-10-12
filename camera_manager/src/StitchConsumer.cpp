#include "StitchConsumer.h"

void StitchConsumer::single_stitch(int cam_id) {
    while(running) {
        Frame tmp;
        if(!m_frame[cam_id]->wait_and_pop(tmp)) break;
        
    }
}

StitchConsumer::StitchConsumer(std::vector<safe_queue<Frame> *> frame_to_stitch, int width, int height)
{
    m_frame = frame_to_stitch;
    m_name += "stitch";
    cam_num = m_frame.size();
    this->width = width;
    this->height = height;
    int output_width = width * cam_num;
    if(config::GetInstance().GetGlobalStitchConfig().output_width != -1) {
        output_width = config::GetInstance().GetGlobalStitchConfig().output_width;
    }
    stitch.init(width, height, cam_num);
    url = config::GetInstance().GetGlobalStitchConfig().output_url;
    avformat_alloc_output_context2(&out_ctx, nullptr, "rtsp", url.c_str());
    
    out_stream = avformat_new_stream(out_ctx, nullptr);
    out_stream->id = out_ctx->nb_streams - 1; // 设置流ID
    out_stream->time_base = (AVRational){1, 20};

    codecpar = out_stream->codecpar;
    codecpar->codec_type = AVMEDIA_TYPE_VIDEO;
    codecpar->codec_id = AV_CODEC_ID_H264;   
    codecpar->width = output_width;                  
    codecpar->height = height;                 
    codecpar->format = AV_PIX_FMT_CUDA;

    m_status.width = codecpar->width;
    m_status.height = codecpar->height;
}

safe_queue<Frame> &StitchConsumer::get_stitch_frame()
{
    return frame_output;
}

StitchConsumer::~StitchConsumer() {

}

void StitchConsumer::start() {
    TaskManager::start();
    for(int i=0;i<cam_num;i++) {
        m_threads[i] = std::thread(&StitchConsumer::single_stitch, this, i);
    }
}

void StitchConsumer::stop() {
    for(auto& i: m_frame) i->stop();
    TaskManager::stop();
    for(auto& thread : m_threads) {
        if(thread.joinable())
            thread.join();
    }
}

void StitchConsumer::run() { 
    Frame out_image;
    AVFrame** inputs = new AVFrame*[cam_num];
    while (running) {
        for (int i = 0; i < cam_num; i++) {
            Frame tmp;
            if(!m_frame[i]->wait_and_pop(tmp)) goto cleanup;
            inputs[i] = tmp.m_data;
            out_image.m_costTimes.image_frame_cnt[i] = tmp.m_costTimes.image_frame_cnt[i];
            out_image.m_costTimes.when_get_packet[i] = tmp.m_costTimes.when_get_packet[i];
            out_image.m_costTimes.when_get_decoded_frame[i] = tmp.m_costTimes.when_get_decoded_frame[i];
        }
        out_image.m_data = stitch.do_stitch(inputs);
        out_image.m_data->pts = inputs[0]->pts;
        out_image.m_costTimes.when_get_stitched_frame = get_now_time();
        frame_output.push(out_image);
        m_status.frame_cnt ++;
        m_status.timestamp = get_now_time();
        for (int i = 0; i < cam_num; ++i) {
            if (inputs[i]) {
                av_frame_free(&inputs[i]);
            }
        }
    }
cleanup:
    for(int i = 0;i < cam_num; i++) {
        m_frame[i]->clear();
    }
    delete[] inputs;
}
