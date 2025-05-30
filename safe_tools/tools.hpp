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

void save_frame_as_nv12(AVFrame* frame, const std::string& filename) {
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

void transfer_and_save_cuda_nv12(AVFrame* hw_frame, const std::string& filename) {
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
