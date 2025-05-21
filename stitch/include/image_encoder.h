
extern "C" {
    #include <libavcodec/avcodec.h>
    #include <libavutil/avutil.h>
    #include <libavutil/hwcontext.h>
    #include <libavformat/avformat.h>
}

#include <string>
#include <stdexcept>

class image_encoder {
public:
    image_encoder(const std::string& codec_name = "h264_nvenc", int width = 19200, int height = 2160, int fps = 20);
    ~image_encoder();

    AVPacket* do_encode(AVFrame* frame);

    AVCodecContext* codec_ctx;
    const AVCodec* codec;
    AVPacket* pkt;
};