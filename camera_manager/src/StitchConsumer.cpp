#include "StitchConsumer.h"

StitchConsumer::StitchConsumer(std::vector<safe_queue<Frame> *> frame_to_stitch, int width, int height) {
    m_frame = frame_to_stitch;
    m_name += "stitch";
    cam_num = m_frame.size();
    this->width = width; // need fix，最好是通过推断或者通过json配置
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

void StitchConsumer::run() { 
    Frame out_image;
    while (running) {
        AVFrame** inputs = new AVFrame*[cam_num];
        for (int i = 0; i < cam_num; i++) {
            Frame tmp;
            m_frame[i]->wait_and_pop(tmp);
            inputs[i] = tmp.m_data;
        }
        out_image.m_data = stitch.do_stitch(inputs);
        out_image.m_data->pts = inputs[0]->pts;
        frame_output.push(out_image);
        m_status.frame_cnt ++;
        for (int i = 0; i < cam_num; ++i) {
            if (inputs[i]) {
                av_frame_free(&inputs[i]);
            }
        }
        delete[] inputs;
    }
}
