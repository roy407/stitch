#pragma once
#include <iostream>
#include <fstream>
extern "C" {
#include <libavutil/frame.h>
#include <libavutil/hwcontext.h>
#include <libswscale/swscale.h>
}

inline void AVFrame_log(const char* cam_name, const AVFrame* frame) {
    if (frame) {
        std::cout << "========= AVFrame Info (" << cam_name << ") =========" << std::endl;
        std::cout << "Format:         " << frame->format << std::endl;
        std::cout << "Width x Height: " << frame->width << " x " << frame->height << std::endl;
        std::cout << "PTS:            " << frame->pts << std::endl;
        std::cout << "DTS:            " << frame->pkt_dts << std::endl;
        std::cout << "HW FramesCtx:   " << (frame->hw_frames_ctx ? "Yes (GPU frame)" : "No (CPU frame)") << std::endl;
        
        // 输出前 3 个 data/buf 指针状态
        for (int i = 0; i < 3; ++i) {
            std::cout << "data[" << i << "]:      " << static_cast<const void*>(frame->data[i]) << std::endl;
            std::cout << "linesize[" << i << "]:  " << frame->linesize[i] << std::endl;
        }

        std::cout << "==========================================================" << std::endl;
    }
}