#pragma once

#include "Producer.h"
#include "LogConsumer.h"
#include <memory>
extern "C" {
    #include "libavformat/avformat.h"
    #include "libavcodec/avcodec.h"
}
#include "config.h"
#include "Channel.h"

class AVFrameProducer : public Producer {
protected:
    PacketChannel* m_channel2rtsp{nullptr};
    PacketChannel* m_channel2decoder{nullptr};
    int cam_id{-1};    // 相机ID，一般是从0开始，依次加一
    AVFormatContext* fmt_ctx{nullptr};
    AVDictionary* options{nullptr};
    std::string cam_path;
    AVRational time_base;
    int video_stream{-1};
    AVCodecParameters* codecpar{nullptr};
    CamStatus m_status{};
    friend class LogConsumer;
public:
    AVFrameProducer(); // for AVFrameProducer_debug, do nothing!
    AVFrameProducer(CameraConfig camera_config);
    AVFrameProducer(int cam_id, std::string name, std::string input_url, int width, int height);
    virtual ~AVFrameProducer();
    virtual void start();
    virtual void stop();
    virtual void run();
    int getWidth() const;
    int getHeight() const;
    AVRational getTimeBase() const;
    AVCodecParameters* getAVCodecParameters() const;
    PacketChannel* getChannel2Rtsp() const;
    PacketChannel* getChannel2Decoder() const;
};