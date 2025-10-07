#pragma once
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
    explicit Stitch();
    void init(int width,int height,int cam_num);
    ~Stitch();
    AVFrame* do_stitch(AVFrame** inputs);
private:
    AVFrame* output;
    AVBufferRef* hw_frames_ctx;
    std::atomic_bool running;
    int cam_num;
    int single_width;
    int height;
    int output_width;
    uint8_t **d_inputs_y;
    uint8_t **d_inputs_uv;
    int* d_input_linesize_y;
    int* d_input_linesize_uv;
    float* d_h_matrix;
    int* d_crop;
    float* h_matrix;
    float** d_cam_polygons;
};