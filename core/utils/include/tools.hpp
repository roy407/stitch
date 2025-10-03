#pragma once
#include <iostream>
#include <fstream>
#include <chrono>
#include <string>
#include <ctime>
#include <iomanip>
#include <sstream>
extern "C" {
#include <libavutil/frame.h>
#include <libavutil/hwcontext.h>
#include <libswscale/swscale.h>
}

struct costTimes {
    uint64_t image_idx[5];
    uint64_t when_get_packet[5];
    uint64_t when_get_decoded_frame[5];
    uint64_t when_get_stitched_frame;
    uint64_t when_show_on_the_screen;
};

struct Frame {
    int cam_id;
    AVFrame* m_data;
    struct costTimes m_costTimes;
};

struct Packet {
    int cam_id;
    AVPacket* m_data;
    struct costTimes m_costTimes;
};

inline uint64_t get_now_time() {
    auto now = std::chrono::system_clock::now();
    auto ns_since_epoch = std::chrono::duration_cast<std::chrono::nanoseconds>(
        now.time_since_epoch()
    ).count();
    return static_cast<uint64_t>(ns_since_epoch);
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

inline std::string get_current_time_filename(const std::string& suffix = ".txt") {
    auto now = std::chrono::system_clock::now();
    std::time_t now_time_t = std::chrono::system_clock::to_time_t(now);
    std::tm now_tm;

#ifdef _WIN32
    localtime_s(&now_tm, &now_time_t);
#else
    localtime_r(&now_time_t, &now_tm);
#endif

    std::ostringstream oss;
    oss << std::put_time(&now_tm, "%Y-%m-%d_%H-%M-%S") << suffix;
    return oss.str();
}
inline void save_cost_times_to_timestamped_file(const costTimes& ct, std::ofstream& ofs) {

    ofs << "=== costTimes Latency Record (毫秒) ===\n";

    // image_idx
    ofs << "image_idx: ";
    for (int i = 0; i < 5; ++i) {
        ofs << i;
        if (i < 4) ofs << ", ";
    }
    ofs << "\n";

    // packet_latency = when_get_packet - image_idx
    ofs << "图像编号: ";
    for (int i = 0; i < 5; ++i) {
        ofs << (ct.image_idx[i]);
        if (i < 4) ofs << ", ";
    }
    ofs << "\n";

    // decode_latency = when_get_decoded_frame - when_get_packet
    ofs << "解码耗时: ";
    for (int i = 0; i < 5; ++i) {
        ofs << (ct.when_get_decoded_frame[i] - ct.when_get_packet[i]) / 1000000;
        if (i < 4) ofs << ", ";
    }
    ofs << "\n";

    // stitch_latency = when_get_stitched_frame - when_get_decoded_frame[0]
    ofs << "拼接耗时: " << (ct.when_get_stitched_frame - ct.when_get_decoded_frame[0]) / 1000000 << "\n";

    // display_latency = when_show_on_the_screen - when_get_stitched_frame
    ofs << "显示耗时: " << (ct.when_show_on_the_screen - ct.when_get_stitched_frame) / 1000000 << "\n";

    ofs << "-------------------------\n";
}

