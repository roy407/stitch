#pragma once

#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>

extern "C" {
    #include <libavcodec/avcodec.h>
    #include <libavformat/avformat.h>
    #include <libavutil/log.h>
    #include <libavutil/opt.h>
}

#include "config.h"


#include "cuda_handle_init.h"

template<int FormatValue>
struct Format {
    static constexpr int value = FormatValue;
    /*
    0: YUV420
    1: YUV420P
    */
};

using YUV420 = Format<0>;
using YUV420P = Format<1>;

static constexpr int H_MATRIX_SIZE = 9;
static constexpr int CAM_POLYGON_SIZE = 8;

//未测试
struct RawKernel {
    bool initgetKernelGpuMemory(int num) { return true; }
    void freeKernelGpuMemory() {}
};
//未测试
struct CropKernel {
    int* d_crop{nullptr};
    std::vector<int> crop_rects;
    void loadCropRects(std::vector<int> rects) {
        crop_rects = rects;
    }
    bool initgetKernelGpuMemory(int num) {
        if (crop_rects.empty()) return false;
        if (cudaMalloc(&d_crop, num * 4 * sizeof(int)) != cudaSuccess) return false;
        if (cudaMemcpy(d_crop, crop_rects.data(), num * 4 * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) return false;
        return true;
    }
    void freeKernelGpuMemory() {
        if (d_crop) { cudaFree(d_crop); d_crop = nullptr; }
    }
};
//未测试
struct HMatrixInvKernel {
    float* d_h_matrix_inv{nullptr};
    float** d_cam_polygons{nullptr};
    std::vector<float> h_matrix_inv;
    std::vector<float> cam_polygons;
    void loadConfig(const StitchImplConfig& cfg, int num) {
        h_matrix_inv.resize(num * H_MATRIX_SIZE);
        for(int i=0; i<num; i++) {
            if (i < cfg.h_matrix_inv.size()) {
                for(int j=0; j<H_MATRIX_SIZE; j++) {
                    h_matrix_inv[i*H_MATRIX_SIZE+j] = static_cast<float>(cfg.h_matrix_inv[i][j]);
                }
            }
        }
        cam_polygons.resize(num * CAM_POLYGON_SIZE);
        for(int i=0; i<num; i++) {
            if (i < cfg.cam_polygons.size()) {
                for(int j=0; j<CAM_POLYGON_SIZE; j++) {
                    cam_polygons[i*CAM_POLYGON_SIZE+j] = static_cast<float>(cfg.cam_polygons[i][j]);
                }
            }
        }
    }
    bool initgetKernelGpuMemory(int num) {
        bool success = true;
        if (!h_matrix_inv.empty()) {
            if (cudaMalloc(&d_h_matrix_inv, sizeof(float) * H_MATRIX_SIZE * num) != cudaSuccess) success = false;
            else if (cudaMemcpy(d_h_matrix_inv, h_matrix_inv.data(), sizeof(float) * H_MATRIX_SIZE * num, cudaMemcpyHostToDevice) != cudaSuccess) success = false;
        }
        if (success && !cam_polygons.empty()) {
            if (cudaMalloc(&d_cam_polygons, sizeof(float*) * num) != cudaSuccess) success = false;
            else {
                float** h_cam_ptrs = new float*[num];
                for (int i = 0; i < num; ++i) {
                    if (cudaMalloc(&h_cam_ptrs[i], sizeof(float) * CAM_POLYGON_SIZE) != cudaSuccess) { success = false; break; }
                    if (cudaMemcpy(h_cam_ptrs[i], &cam_polygons[i * CAM_POLYGON_SIZE], sizeof(float) * CAM_POLYGON_SIZE, cudaMemcpyHostToDevice) != cudaSuccess) { success = false; break; }
                }
                if (success) {
                    if (cudaMemcpy(d_cam_polygons, h_cam_ptrs, sizeof(float*) * num, cudaMemcpyHostToDevice) != cudaSuccess) success = false;
                }
                delete[] h_cam_ptrs;
            }
        }
        return success;
    }
    void freeKernelGpuMemory() {
        if (d_h_matrix_inv) { cudaFree(d_h_matrix_inv); d_h_matrix_inv = nullptr; }
        if (d_cam_polygons) { cudaFree(d_cam_polygons); d_cam_polygons = nullptr; }
    }
};
//未测试
struct HMatrixInvV1_1Kernel {
    float* d_h_matrix_inv{nullptr};
    float** d_cam_polygons{nullptr};
    std::vector<float> h_matrix_inv;
    std::vector<float> cam_polygons;
    void loadConfig(const StitchImplConfig& cfg, int num) {
        h_matrix_inv.resize(num * H_MATRIX_SIZE);
        for(int i=0; i<num; i++) {
            if (i < cfg.h_matrix_inv.size()) {
                for(int j=0; j<H_MATRIX_SIZE; j++) {
                    h_matrix_inv[i*H_MATRIX_SIZE+j] = static_cast<float>(cfg.h_matrix_inv[i][j]);
                }
            }
        }
        cam_polygons.resize(num * CAM_POLYGON_SIZE);
        for(int i=0; i<num; i++) {
            if (i < cfg.cam_polygons.size()) {
                for(int j=0; j<CAM_POLYGON_SIZE; j++) {
                    cam_polygons[i*CAM_POLYGON_SIZE+j] = static_cast<float>(cfg.cam_polygons[i][j]);
                }
            }
        }
    }
    bool initgetKernelGpuMemory(int num) {
        bool success = true;
        if (!h_matrix_inv.empty()) {
            if (cudaMalloc(&d_h_matrix_inv, sizeof(float) * H_MATRIX_SIZE * num) != cudaSuccess) success = false;
            else if (cudaMemcpy(d_h_matrix_inv, h_matrix_inv.data(), sizeof(float) * H_MATRIX_SIZE * num, cudaMemcpyHostToDevice) != cudaSuccess) success = false;
        }
        if (success && !cam_polygons.empty()) {
            if (cudaMalloc(&d_cam_polygons, sizeof(float*) * num) != cudaSuccess) success = false;
            else {
                float** h_cam_ptrs = new float*[num];
                for (int i = 0; i < num; ++i) {
                    if (cudaMalloc(&h_cam_ptrs[i], sizeof(float) * CAM_POLYGON_SIZE) != cudaSuccess) { success = false; break; }
                    if (cudaMemcpy(h_cam_ptrs[i], &cam_polygons[i * CAM_POLYGON_SIZE], sizeof(float) * CAM_POLYGON_SIZE, cudaMemcpyHostToDevice) != cudaSuccess) { success = false; break; }
                }
                if (success) {
                    if (cudaMemcpy(d_cam_polygons, h_cam_ptrs, sizeof(float*) * num, cudaMemcpyHostToDevice) != cudaSuccess) success = false;
                }
                delete[] h_cam_ptrs;
            }
        }
        return success;
    }
    void freeKernelGpuMemory() {
        if (d_h_matrix_inv) { cudaFree(d_h_matrix_inv); d_h_matrix_inv = nullptr; }
        if (d_cam_polygons) { cudaFree(d_cam_polygons); d_cam_polygons = nullptr; }
    }
};
struct HMatrixInvV2Kernel {
    float* d_h_matrix_inv{nullptr};
    std::vector<float> h_matrix_inv;
    void loadConfig(const StitchImplConfig& cfg, int num) {
        h_matrix_inv.resize(num * H_MATRIX_SIZE);
        for(int i=0; i<num; i++) {
            if (i < cfg.h_matrix_inv.size()) {
                for(int j=0; j<H_MATRIX_SIZE; j++) {
                    h_matrix_inv[i*H_MATRIX_SIZE+j] = static_cast<float>(cfg.h_matrix_inv[i][j]);
                }
            }
        }
    }
    bool initgetKernelGpuMemory(int num) {
        if (h_matrix_inv.empty()) return false;
        if (cudaMalloc(&d_h_matrix_inv, sizeof(float) * H_MATRIX_SIZE * num) != cudaSuccess) return false;
        if (cudaMemcpy(d_h_matrix_inv, h_matrix_inv.data(), sizeof(float) * H_MATRIX_SIZE * num, cudaMemcpyHostToDevice) != cudaSuccess) return false;
        return true;
    }
    void freeKernelGpuMemory() {
        if (d_h_matrix_inv) { cudaFree(d_h_matrix_inv); d_h_matrix_inv = nullptr; }
    }
};

struct MappingTableKernel {
    cudaTextureObject_t d_mapping_table{0};
    void loadMappingTable(cudaTextureObject_t tex) {
        d_mapping_table = tex;
    }
    bool initgetKernelGpuMemory(int num) { return true; }
    void freeKernelGpuMemory() {}
};

template <typename Derived>
class IStitch {
public:
    ~IStitch() {
        if (hw_frames_ctx) {
            av_buffer_unref(&hw_frames_ctx);
        }
    };

    void init(int num, int single_width, int output_width, int height) {
        this->num = num;
        this->single_width = single_width;
        this->output_width = output_width;
        this->height = height;
        LOG_DEBUG("stitch initing ... cam_num : {} , width : {} , stitched_width : {} , height : {}", num, single_width, output_width, height);
        CreateHWFramesCtx();
        static_cast<Derived*>(this)->init_impl();
    }

    AVFrame* do_stitch(AVFrame** inputs) {
        return static_cast<Derived*>(this)->do_stitch_impl(inputs);
    }

protected:
    AVFrame* output{nullptr};
    AVBufferRef* hw_frames_ctx{nullptr};

    int num = 0;
    int single_width = 0;
    int output_width = 0;
    int height = 0;

    bool CreateHWFramesCtx();
};

// 创建 HW frame context
template <typename Derived>
bool IStitch<Derived>::CreateHWFramesCtx() {
    hw_frames_ctx = av_hwframe_ctx_alloc(cuda_handle_init::GetGPUDeviceHandle());
    AVHWFramesContext* frames_ctx = (AVHWFramesContext*)hw_frames_ctx->data;
    frames_ctx->format = AV_PIX_FMT_CUDA;
    frames_ctx->sw_format = AV_PIX_FMT_NV12;   // CUDA 支持的底层格式
    frames_ctx->width = output_width;
    frames_ctx->height = height;
    frames_ctx->initial_pool_size = 20;

    if (av_hwframe_ctx_init(hw_frames_ctx) < 0) {
        throw std::runtime_error("Failed to initialize CUDA hwframe context");
    }
    return true;
}