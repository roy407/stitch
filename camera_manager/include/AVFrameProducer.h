#pragma once

#include "Producer.h"
#include "Consumer.h"
#include "StitchConsumer.h"
#include "LogConsumer.h"
#include "image_decoder.h"
#include <memory>
extern "C" {
    #include "libavformat/avformat.h"
    #include "libavcodec/avcodec.h"
    #include "libavutil/pixfmt.h" 
    #include "libavutil/pixdesc.h" 
    #include "libavutil/opt.h"
    #include "libavutil/log.h"
    #include "libavcodec/bsf.h"
}
#include "config.h"

class AVFrameProducer : public Producer {
    safe_queue<AVFrame*> m_frameSender;
    safe_queue<AVPacket*> m_packetSender1; // rtsp
    safe_queue<AVPacket*> m_packetSender2; // decoder
    image_decoder img_dec;
    int cam_id{-1};    // 相机ID，一般是从0开始，依次加一
    AVFormatContext* fmt_ctx{nullptr};
    AVDictionary* options{nullptr};
    std::string cam_path;
    AVStream* stream{nullptr};
    int video_stream{-1};
    AVCodecParameters* codecpar{nullptr};
    CamStatus m_status{};
public:
    AVFrameProducer(int cam_id);
    virtual ~AVFrameProducer();
    virtual void start();
    virtual void stop();
    virtual void run();
    int getWidth() const;
    int getHeight() const;
    safe_queue<AVFrame*>& getFrameSender();
    safe_queue<AVPacket*>& getPacketSender();
private:
    std::unique_ptr<TaskManager> m_rtspConsumer; // 单个相机的推流线程，自己创建
    friend class LogConsumer;
};