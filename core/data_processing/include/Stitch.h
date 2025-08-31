#pragma once
#include <thread>
#include <atomic>
#include <acl/acl.h>
extern "C" {
    #include <libavformat/avformat.h>
    #include <libavcodec/avcodec.h>
    #include <libavutil/opt.h>
    #include <libavutil/log.h>
}
class Stitch {
public:
    explicit Stitch(int width,int height,int cam_num);
    ~Stitch();
    AVFrame* do_stitch(AVFrame** inputs);
private:
    AVFrame* output;
    AVBufferRef* hw_frames_ctx;
    aclrtStream stream;
    std::atomic_bool running;
    const int cam_num;
    const int single_width;
    const int height;
    int output_width;
    uint8_t **d_inputs_y;
    uint8_t **d_inputs_uv;
    int* d_input_linesize_y;
    int* d_input_linesize_uv;
    float* d_h_matrices;
    int* d_crop;
};