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
    std::atomic_bool running;
    AVFrame* output;
    AVBufferRef* hw_frames_ctx;
    bool CreateHWFramesCtx();
    int cam_num;
    int single_width;
    int height;
    int output_width;
    bool SetCameraAttribute(int width, int height, int cam_num);
// 下面的代码都是跟gpu有关
    uint8_t **d_inputs_y{nullptr};
    uint8_t **d_inputs_uv{nullptr};
    int* d_input_linesize_y{nullptr};
    int* d_input_linesize_uv{nullptr};
    bool AllocateFrameBufPtr();
    bool MemoryCpyFrameBufPtr(AVFrame** inputs);
    int* d_crop{nullptr};
    bool SetCrop();
    float* d_h_matrix_inv{nullptr};
    bool SetHMatrixInv();
    float** d_cam_polygons{nullptr};
    bool SetCamPolygons();
    const uint16_t* d_mapping_table{nullptr};
    bool LoadMappingTable();
};