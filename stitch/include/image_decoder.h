
extern "C" {
    #include <libavcodec/avcodec.h>
    #include <libavutil/avutil.h>
    #include <libavutil/hwcontext.h>
    #include <libavformat/avformat.h>
}

#include <iostream>
#include <queue>
#include <stdexcept>
#include <atomic>

class image_decoder {
public:
    image_decoder(const std::string& codec_name = "h264_cuvid");
    ~image_decoder();
    void set_parameter(AVCodecParameters* codecpar);
    std::queue<AVFrame*> do_decode(const AVPacket* pkt);

    AVCodecContext* codec_ctx;
    const AVCodec* codec;
    AVFrame* frame;
    std::atomic_bool is_created;
};
