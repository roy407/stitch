#include "Stitch.h"
#include "stitch_with_h_matrix_inv.cuh"
#include "scale.cuh"
#include <iostream>

#include <libavutil/frame.h>
#include <libavutil/hwcontext.h>
#include <libavutil/pixfmt.h>

#include "cuda_handle_init.h"
#include "config.h"

Stitch::Stitch() {
    
}

void Stitch::init(int width, int height, int cam_num) {
    SetCameraAttribute(width, height, cam_num);
    AllocateFrameBufPtr();
    #if defined(LAUNCH_STITCH_KERNEL_WITH_CROP)
        SetCrop();
    #endif
    #if defined(LAUNCH_STITCH_KERNEL_WITH_H_MATRIX_INV)
        SetHMatrixInv();
        SetCamPolygons();
    #endif
    LoadMappingTable();
    CreateHWFramesCtx();
}

Stitch::~Stitch() {
    cudaFree(d_inputs_y);
    cudaFree(d_inputs_uv);
    cudaFree(d_crop);
    cudaFree(d_input_linesize_y);
    cudaFree(d_input_linesize_uv);
    cudaFree(d_h_matrix_inv);
}

AVFrame* Stitch::do_stitch(AVFrame** inputs) {
    MemoryCpyFrameBufPtr(inputs);

    output = av_frame_alloc();
    if (!output) {
        throw std::runtime_error("Failed to allocate output frame");
    }
    output->format = AV_PIX_FMT_CUDA;
    output->width = output_width;
    output->height = height;
    output->hw_frames_ctx = av_buffer_ref(hw_frames_ctx);
    if (av_hwframe_get_buffer(hw_frames_ctx, output, 0) < 0) {
        throw std::runtime_error("Failed to allocate GPU AVFrame buffer");
    }
    uint8_t* output_y = output->data[0];
    uint8_t* output_uv = output->data[1];
    cudaStream_t stream = 0; // 修改，使用五个流，同时拼。

    #if defined(LAUNCH_STITCH_KERNEL_RAW)
        launch_stitch_kernel_raw(d_inputs_y, d_inputs_uv,
            d_input_linesize_y, d_input_linesize_uv,
            output_y, output_uv,
            output->linesize[0], output->linesize[1],
            cam_num, single_width, output_width, height,stream);
    #elif defined(LAUNCH_STITCH_KERNEL_WITH_CROP)
        launch_stitch_kernel_with_crop(d_inputs_y, d_inputs_uv,
            d_input_linesize_y, d_input_linesize_uv,
            output_y, output_uv,
            output->linesize[0], output->linesize[1],
            cam_num, single_width, output_width, height, stream, d_crop);
    #elif defined(LAUNCH_STITCH_KERNEL_WITH_H_MATRIX_INV)
        launch_stitch_kernel_with_h_matrix_inv(d_inputs_y, d_inputs_uv,
            d_input_linesize_y, d_input_linesize_uv,
            d_h_matrix_inv, d_cam_polygons,
            output_y, output_uv,
            output->linesize[0], output->linesize[1],
            cam_num, single_width, output_width, height,stream);
    #elif defined(LAUNCH_STITCH_KERNEL_WITH_MAPPING_TABLE)
        launch_stitch_kernel_with_mapping_table(d_inputs_y, d_inputs_uv,
            d_input_linesize_y, d_input_linesize_uv,
            output_y, output_uv,
            output->linesize[0], output->linesize[1],
            cam_num, single_width, output_width, height, d_mapping_table, stream);
    #else
        static int __cnt__ = 0;
        __cnt__ ++;
        if(__cnt__ % 20 == 0) LOG_WARN("No kernel is running now");
    #endif
    cudaStreamSynchronize(stream);
    return output;
}

bool Stitch::CreateHWFramesCtx() {
    // 创建 HW frame context
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

bool Stitch::SetCameraAttribute(int width, int height, int cam_num) {
    this->single_width = width;
    this->height = height;
    this->cam_num = cam_num;
    output_width = single_width * cam_num;
    if(config::GetInstance().GetGlobalStitchConfig().output_width != -1) {
        output_width = config::GetInstance().GetGlobalStitchConfig().output_width;
    }
    return true;
}

bool Stitch::AllocateFrameBufPtr() {
    CHECK_CUDA(cudaMalloc(&d_inputs_y, sizeof(uint8_t*) * cam_num));
    CHECK_CUDA(cudaMalloc(&d_inputs_uv, sizeof(uint8_t*) * cam_num));
    CHECK_CUDA(cudaMalloc(&d_input_linesize_y, cam_num * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_input_linesize_uv, cam_num * sizeof(int)));
    return true;
}

bool Stitch::MemoryCpyFrameBufPtr(AVFrame** inputs) {
    uint8_t* gpu_inputs_y[cam_num];
    uint8_t* gpu_inputs_uv[cam_num];
    for (int i = 0; i < cam_num; ++i) {
        if (!inputs[i]) {
            return false;
        }
        gpu_inputs_y[i] = inputs[i]->data[0];
        gpu_inputs_uv[i] = inputs[i]->data[1];
    }

    cudaMemcpy(d_inputs_y, gpu_inputs_y, sizeof(uint8_t*) * cam_num, cudaMemcpyHostToDevice);
    cudaMemcpy(d_inputs_uv, gpu_inputs_uv, sizeof(uint8_t*) * cam_num, cudaMemcpyHostToDevice);
    /*如果是不变量，考虑只初始化一次---待修改*/
    int h_input_linesize_y[cam_num];
    int h_input_linesize_uv[cam_num];

    for(int i=0;i<cam_num;i++) {
        h_input_linesize_uv[i] = inputs[i]->linesize[1];
        h_input_linesize_y[i] = inputs[i]->linesize[0];
    }

    cudaMemcpy(d_input_linesize_y, h_input_linesize_y, cam_num * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input_linesize_uv, h_input_linesize_uv, cam_num * sizeof(int), cudaMemcpyHostToDevice);
    return true;
}

bool Stitch::SetCrop() {
    int* crop = new int[cam_num * 4];
    memset(crop,0,cam_num * 4);
    CHECK_CUDA(cudaMalloc(&d_crop, cam_num * 4 * sizeof(int)));
    const std::vector<CameraConfig> cams = config::GetInstance().GetCameraConfig();
    for(int i=0;i<cam_num;i++) {
        if(cams[i].stitch.enable == true) {
            if(cams[i].stitch.mode == "crop") {
                std::vector<float> __crop = cams[i].crop;
                crop[i*4] = __crop[0] * single_width;
                crop[i*4+1] = __crop[1] * height;
                crop[i*4+2] = __crop[2] * single_width;
                crop[i*4+3] = __crop[3] * height;
            }
        }
    }
    CHECK_CUDA(cudaMemcpy(d_crop, crop, cam_num * sizeof(int), cudaMemcpyHostToDevice));
    delete[] crop;
    return true;
}

bool Stitch::SetHMatrixInv() {
    CHECK_CUDA(cudaMalloc(&d_h_matrix_inv, sizeof(float) * 9 * cam_num));
    const std::vector<std::array<double, 9>> __h_matrix_inv = config::GetInstance().GetGlobalStitchConfig().h_matrix_inv;
    float* h_matrix_inv = new float[cam_num * 9];
    for(int i=0;i<cam_num;i++) {
        for(int j=0;j<9;j++) {
            h_matrix_inv[i*9+j] = static_cast<float>(__h_matrix_inv[i][j]);
        }
    }
    CHECK_CUDA(cudaMemcpy(d_h_matrix_inv, h_matrix_inv, sizeof(float) * 9 * cam_num, cudaMemcpyHostToDevice));
    delete[] h_matrix_inv;
    return true;
}

bool Stitch::SetCamPolygons() {
    const std::vector<std::array<float, 8>> cam_polygons = config::GetInstance().GetGlobalStitchConfig().cam_polygons;
    CHECK_CUDA(cudaMalloc(&d_cam_polygons, sizeof(float*) * cam_num));
    float** h_cam_ptrs = new float*[cam_num];
    for (int i = 0; i < cam_num; ++i) {
        CHECK_CUDA(cudaMalloc(&h_cam_ptrs[i], sizeof(float) * 8));
        CHECK_CUDA(cudaMemcpy(h_cam_ptrs[i], cam_polygons[i].data(), sizeof(float)*8, cudaMemcpyHostToDevice));
    }

    CHECK_CUDA(cudaMemcpy(d_cam_polygons, h_cam_ptrs,
               sizeof(float*) * cam_num, cudaMemcpyHostToDevice));
    delete[] h_cam_ptrs;
    return true;
}

bool Stitch::LoadMappingTable() {
    d_mapping_table = config::GetInstance().GetMappingTable();
    return true;
}
