#include <thread>
#include <atomic>
extern "C" {
    #include <libavformat/avformat.h>
    #include <libavcodec/avcodec.h>
    #include <libavutil/opt.h>
    #include <libavutil/log.h>
}
class Stitch {
public:
    explicit Stitch(int cam_num,int single_width,int height);
    ~Stitch();
    AVFrame* do_stitch(AVFrame** inputs);
private:
    AVFrame* output;
    AVBufferRef* hw_frames_ctx;
    size_t size;
    std::atomic_bool running;
    int cam_num;
    int single_width;
    int height;
};