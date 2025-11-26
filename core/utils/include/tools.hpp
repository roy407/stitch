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

struct costTimes {
    uint64_t image_frame_cnt[10];
    uint64_t when_get_packet[10];
    uint64_t when_get_decoded_frame[10];
    uint64_t when_get_stitched_frame;
    uint64_t when_show_on_the_screen;
};

struct Frame {
    int cam_id;
    AVFrame* m_data;
    struct costTimes m_costTimes;
    uint64_t m_timestamp;
};

struct Packet {
    int cam_id;
    AVPacket* m_data;
    struct costTimes m_costTimes;
    uint64_t m_timestamp;
};

uint64_t get_now_time();
std::string get_current_time_filename(const std::string& suffix = ".txt");

// === NV12 存储 ===
void save_frame_as_nv12(AVFrame* frame, const std::string& filename);
void transfer_and_save_cuda_nv12(AVFrame* hw_frame, const std::string& filename);

// === 性能统计 ===
void save_cost_times_to_timestamped_file(const costTimes& t, std::ofstream& ofs);
void save_cost_table_csv(const costTimes& t, std::ofstream& ofs);

// === 绘图工具 ===
void draw_vertical_line_nv12(AVFrame *frame, int x, const std::string label, int fst, int Y);
