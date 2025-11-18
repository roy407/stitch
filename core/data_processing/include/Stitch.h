#pragma once
#include <thread>
#include <atomic>
#include <cuda_runtime.h>
extern "C" {
    #include <libavformat/avformat.h>
    #include <libavcodec/avcodec.h>
    #include <libavutil/opt.h>
    #include <libavutil/log.h>
}
#include "h_matrix_inv/stitch_with_h_matrix_inv_v2.cuh"

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
    #if !defined(LAUNCH_STITCH_KERNEL_WITH_MAPPING_TABLE_YUV420P)
        uint8_t **d_inputs_y{nullptr};
        uint8_t **d_inputs_uv{nullptr};
        int* d_input_linesize_y{nullptr};
        int* d_input_linesize_uv{nullptr};
        bool AllocateFrameBufPtrYUV420();
        bool MemoryCpyFrameBufPtrYUV420(AVFrame** inputs);
    #else
        uint8_t **d_inputs_y{nullptr};
        uint8_t **d_inputs_u{nullptr};
        uint8_t **d_inputs_v{nullptr};
        int* d_input_linesize_y{nullptr};
        int* d_input_linesize_u{nullptr};
        int* d_input_linesize_v{nullptr};
        bool AllocateFrameBufPtrYUV420P();
        bool MemoryCpyFrameBufPtrYUV420P(AVFrame** inputs);
    #endif
    #if defined(LAUNCH_STITCH_KERNEL_WITH_CROP)
        int* d_crop{nullptr};
        bool SetCrop();
    #endif
    #if defined(ENABLE_H_MATRIX_INV)
        float* d_h_matrix_inv{nullptr};
        bool SetHMatrixInv();
        float** d_cam_polygons{nullptr};
        bool SetCamPolygons();
    #endif
    cudaTextureObject_t d_mapping_table{0};
    bool LoadMappingTable();
};