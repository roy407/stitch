
extern "C" {
    #include <libavcodec/avcodec.h>
    #include <libavutil/avutil.h>
    #include <libavutil/hwcontext.h>
    #include <libavformat/avformat.h>
}

#include <string>
#include <stdexcept>
#include <atomic>

class image_encoder {
public:
    image_encoder(const std::string& codec_name = "h264_nvenc");
    ~image_encoder();
    void start_image_encoder();
    void close_image_encoder();

    AVPacket* do_encode(AVFrame* frame);

    AVCodecContext* codec_ctx;
    const AVCodec* codec;
    AVPacket* pkt;
    std::atomic_bool is_created;
};