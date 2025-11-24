#include "tools.hpp"

// ------------------------
// 获取当前 ns 时间
// ------------------------
uint64_t get_now_time() {
    auto now = std::chrono::system_clock::now();
    auto ns_since_epoch = std::chrono::duration_cast<std::chrono::nanoseconds>(
        now.time_since_epoch()
    ).count();
    return static_cast<uint64_t>(ns_since_epoch);
}

// ------------------------
// 保存 CPU 上的 NV12 帧
// ------------------------
void save_frame_as_nv12(AVFrame* frame, const std::string& filename) {
    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    int width  = frame->width;
    int height = frame->height;

    // 写 Y 平面
    for (int y = 0; y < height; ++y) {
        ofs.write(reinterpret_cast<char*>(frame->data[0] + y * frame->linesize[0]), width);
    }

    // 写 UV（NV12 → UV 高度 = height/2）
    for (int y = 0; y < height / 2; ++y) {
        ofs.write(reinterpret_cast<char*>(frame->data[1] + y * frame->linesize[1]), width);
    }
}

// ------------------------
// 把 CUDA frame 转到 CPU 并保存成 NV12
// ------------------------
void transfer_and_save_cuda_nv12(AVFrame* hw_frame, const std::string& filename) {
    AVFrame* cpu_frame = av_frame_alloc();

    if (av_hwframe_transfer_data(cpu_frame, hw_frame, 0) < 0) {
        av_frame_free(&cpu_frame);
        throw std::runtime_error("Failed to transfer frame to CPU");
    }

    save_frame_as_nv12(cpu_frame, filename);
    av_frame_free(&cpu_frame);
}

// ------------------------
// 生成带时间戳的文件名
// ------------------------
std::string get_current_time_filename(const std::string& suffix) {
    auto now = std::chrono::system_clock::now();
    std::time_t now_time_t = std::chrono::system_clock::to_time_t(now);
    
    std::tm now_tm{};
#ifdef _WIN32
    localtime_s(&now_tm, &now_time_t);
#else
    localtime_r(&now_time_t, &now_tm);
#endif

    std::ostringstream oss;
    oss << std::put_time(&now_tm, "%Y-%m-%d_%H-%M-%S") << suffix;
    return oss.str();
}

// ------------------------
// 打印表格（txt）格式
// ------------------------
void save_cost_times_to_timestamped_file(const costTimes& t, std::ofstream& ofs) {
    constexpr double scale = 1e-6; // ns -> ms（若你的时间戳是std::chrono::steady_clock::now().time_since_epoch().count()）
    // 如果是直接用毫秒时间戳，请改成 scale = 1.0;

    ofs << std::fixed << std::setprecision(3);
    ofs << "\n================= Pipeline Cost Table =================\n";
    ofs << std::setw(10) << "Camera"
              << std::setw(15) << "Pkt->Dec (ms)"
              << std::setw(15) << "Dec->Stitch (ms)"
              << std::setw(15) << "Stitch->Show (ms)"
              << std::setw(15) << "Total (ms)"
              << std::endl;

    ofs << "--------------------------------------------------------\n";

    for (int i = 0; i < 10; ++i) {
        if (t.when_get_packet[i] == 0 || t.when_get_decoded_frame[i] == 0)
            continue;

        double pkt_to_dec = (t.when_get_decoded_frame[i] - t.when_get_packet[i]) * scale;
        double dec_to_stitch = (t.when_get_stitched_frame - t.when_get_decoded_frame[i]) * scale;
        double stitch_to_show = (t.when_show_on_the_screen - t.when_get_stitched_frame) * scale;
        double total = pkt_to_dec + dec_to_stitch + stitch_to_show;

        ofs << std::setw(10) << ("cam_" + std::to_string(i))
                  << std::setw(15) << pkt_to_dec
                  << std::setw(15) << dec_to_stitch
                  << std::setw(15) << stitch_to_show
                  << std::setw(15) << total
                  << std::endl;
    }

    ofs << "========================================================\n";
}

void save_cost_table_csv(const costTimes& t, std::ofstream& ofs) {
    constexpr double scale = 1e-6; // 纳秒→毫秒；若你原始时间戳是毫秒，请改成 1.0
    static bool isWriteHeader = false;

    if(!isWriteHeader) {
        ofs << "Camera,FrameCount,Pkt->Dec(ms),Dec->Stitch(ms),Stitch->Show(ms),Total(ms)\n";
        isWriteHeader = true;
    }

    for (int i = 0; i < 10; ++i) {
        if (t.when_get_packet[i] == 0 || t.when_get_decoded_frame[i] == 0)
            continue;

        double pkt_to_dec = (t.when_get_decoded_frame[i] - t.when_get_packet[i]) * scale;
        double dec_to_stitch = (t.when_get_stitched_frame - t.when_get_decoded_frame[i]) * scale;
        double stitch_to_show = (t.when_show_on_the_screen - t.when_get_stitched_frame) * scale;
        double total = pkt_to_dec + dec_to_stitch + stitch_to_show;

        ofs << "cam_" << i << ','
            << t.image_frame_cnt[i] << ','
            << std::fixed << std::setprecision(3)
            << pkt_to_dec << ','
            << dec_to_stitch << ','
            << stitch_to_show << ','
            << total << '\n';
    }
}