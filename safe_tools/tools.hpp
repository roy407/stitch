#pragma once

#include <iostream>
#include <fstream>
extern "C" {
#include <libavutil/frame.h>
#include <libavutil/hwcontext.h>
#include <libswscale/swscale.h>
}

#include <iostream>
#include <fstream>
extern "C" {
#include <libavutil/frame.h>
#include <libavutil/hwcontext.h>
}

inline void save_frame_as_nv12(AVFrame* frame, const std::string& filename) {
    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }
    int width = frame->width;
    int height = frame->height;

    // Y平面，只写width长度
    for (int y = 0; y < height ; ++y) {
        ofs.write(reinterpret_cast<char*>(frame->data[0] + y * frame->linesize[0]), width);
    }
    // UV平面，只写width长度（UV高度是height/2）
    for (int y = 0; y < height / 2; ++y) {
        ofs.write(reinterpret_cast<char*>(frame->data[1] + y * frame->linesize[1]), width);
    }
    ofs.close();
}

inline void transfer_and_save_cuda_nv12(AVFrame* hw_frame, const std::string& filename) {
    // Step 1: Transfer to CPU
    AVFrame* cpu_frame = av_frame_alloc();
    if (av_hwframe_transfer_data(cpu_frame, hw_frame, 0) < 0) {
        throw std::runtime_error("Failed to transfer frame to CPU");
    }

    // Step 2: Save NV12 raw
    save_frame_as_nv12(cpu_frame, filename);

    // Cleanup
    av_frame_free(&cpu_frame);
}

inline AVBufferRef* create_cuda_hwframe_ctx(AVCodecContext* codec_ctx, int width, int height) {
    AVBufferRef* hw_device_ctx = nullptr;
    if (av_hwdevice_ctx_create(&hw_device_ctx, AV_HWDEVICE_TYPE_CUDA, nullptr, nullptr, 0) < 0) {
        throw std::runtime_error("Failed to create CUDA device context");
    }

    AVBufferRef* frames_ref = av_hwframe_ctx_alloc(hw_device_ctx);
    if (!frames_ref) {
        av_buffer_unref(&hw_device_ctx);
        throw std::runtime_error("Failed to allocate CUDA hwframe context");
    }

    AVHWFramesContext* frames_ctx = (AVHWFramesContext*)frames_ref->data;
    frames_ctx->format = AV_PIX_FMT_CUDA;
    frames_ctx->sw_format = AV_PIX_FMT_NV12;
    frames_ctx->width = width;
    frames_ctx->height = height;
    frames_ctx->initial_pool_size = 10;

    if (av_hwframe_ctx_init(frames_ref) < 0) {
        av_buffer_unref(&frames_ref);
        av_buffer_unref(&hw_device_ctx);
        throw std::runtime_error("Failed to initialize CUDA hwframe context");
    }

    av_buffer_unref(&hw_device_ctx);
    return frames_ref;
}

inline AVFrame* load_nv12_to_gpu_frame(const std::string& filename, AVBufferRef* hw_frames_ctx, int width, int height) {

    // 1. 打开文件
    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs) {
        throw std::runtime_error("Failed to open NV12 file: " + filename);
    }

    int y_size = width * height;
    int uv_size = width * height / 2;
    int total_size = y_size + uv_size;

    std::vector<uint8_t> buffer(total_size);
    ifs.read(reinterpret_cast<char*>(buffer.data()), total_size);
    if (ifs.gcount() != total_size) {
        throw std::runtime_error("Failed to read complete NV12 data");
    }

    // 2. 分配 CPU frame
    AVFrame* cpu_frame = av_frame_alloc();
    cpu_frame->format = AV_PIX_FMT_NV12;
    cpu_frame->width = width;
    cpu_frame->height = height;

    if (av_frame_get_buffer(cpu_frame, 32) < 0) {
        av_frame_free(&cpu_frame);
        throw std::runtime_error("Failed to allocate CPU frame buffer");
    }

    // 填充 Y 平面
    for (int y = 0; y < height; ++y) {
        memcpy(cpu_frame->data[0] + y * cpu_frame->linesize[0],
               buffer.data() + y * width,
               width);
    }

    // 填充 UV 平面
    const uint8_t* uv_src = buffer.data() + y_size;
    for (int y = 0; y < height / 2; ++y) {
        memcpy(cpu_frame->data[1] + y * cpu_frame->linesize[1],
               uv_src + y * width,
               width);
    }

    // 3. 分配 GPU frame
    AVFrame* gpu_frame = av_frame_alloc();
    gpu_frame->format = AV_PIX_FMT_CUDA;  // 注意：实际要用 NV12 对应的 CUDA 格式，可能需要检查
    gpu_frame->width = width;
    gpu_frame->height = height;
    gpu_frame->hw_frames_ctx = av_buffer_ref(hw_frames_ctx);

    if (av_hwframe_get_buffer(hw_frames_ctx, gpu_frame, 0) < 0) {
        av_frame_free(&cpu_frame);
        av_frame_free(&gpu_frame);
        throw std::runtime_error("Failed to allocate GPU frame buffer");
    }

    // 4. 拷贝数据到 GPU
    if (av_hwframe_transfer_data(gpu_frame, cpu_frame, 0) < 0) {
        av_frame_free(&cpu_frame);
        av_frame_free(&gpu_frame);
        throw std::runtime_error("Failed to transfer data to GPU frame");
    }

    av_frame_free(&cpu_frame);
    return gpu_frame;
}


