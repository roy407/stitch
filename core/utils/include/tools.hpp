#pragma once
#include <iostream>
#include <fstream>
#include <chrono>
#include <string>
#include <ctime>
#include <iomanip>
#include <sstream>
extern "C" {
    #include "libavformat/avformat.h"
    #include <libavutil/frame.h>
    #include <libavutil/hwcontext.h>
    #include <libswscale/swscale.h>
}
#include "log.hpp"

constexpr uint32_t MAX_CAM_SIZE = 10;

struct costTimes {
    uint64_t image_frame_cnt[MAX_CAM_SIZE] = {};
    uint64_t when_get_packet[MAX_CAM_SIZE] = {};
    uint64_t when_get_decoded_frame[MAX_CAM_SIZE] = {};
    uint64_t when_get_stitched_frame = 0;
    uint64_t when_show_on_the_screen = 0;
};

struct Frame {
    int cam_id = 0;
    AVFrame* m_data = nullptr;
    struct costTimes m_costTimes;
    uint64_t m_timestamp = 0;
};

struct Packet {
    int cam_id = 0;
    AVPacket* m_data = nullptr;
    struct costTimes m_costTimes;
    uint64_t m_timestamp = 0;
};

uint64_t get_now_time();
std::string get_current_time_filename(const std::string& suffix = ".txt");
AVPixelFormat transfer_string_2_AVPixelFormat(std::string format);

// === NV12 存储 ===
void save_frame_as_nv12(AVFrame* frame, const std::string& filename);
void transfer_and_save_cuda_nv12(AVFrame* hw_frame, const std::string& filename);

// === AVFrame 创建 ===
AVFrame* get_frame_on_cpu_memory(std::string format, int width, int height);
AVFrame* get_frame_on_gpu_memory(std::string format, int width, int height, AVBufferRef* av_buffer);

// === 性能统计 ===
void save_cost_times_to_timestamped_file(const costTimes& t, std::ofstream& ofs);
void save_cost_table_csv(const costTimes& t, std::ofstream& ofs);
void printCostTimes(const costTimes& c);

// === 绘图工具 ===
void draw_vertical_line_nv12(AVFrame *frame, int x, const std::string label, int fst, int Y);
