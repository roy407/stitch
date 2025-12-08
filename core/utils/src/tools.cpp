#include "tools.hpp"

// ------------------------
// 获取当前 ns 时间
// ------------------------
uint64_t get_now_time() {
    auto now = std::chrono::steady_clock::now();
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

void printCostTimes(const costTimes &c) { {
    std::cout << "========== costTimes ==========\n";

    std::cout << "image_frame_cnt: ";
    for (int i = 0; i < 10; ++i) {
        std::cout << c.image_frame_cnt[i] << " ";
    }
    std::cout << "\n";

    std::cout << "when_get_packet: ";
    for (int i = 0; i < 10; ++i) {
        std::cout << c.when_get_packet[i] << " ";
    }
    std::cout << "\n";

    std::cout << "when_get_decoded_frame: ";
    for (int i = 0; i < 10; ++i) {
        std::cout << c.when_get_decoded_frame[i] << " ";
    }
    std::cout << "\n";

    std::cout << "when_get_stitched_frame: "
              << c.when_get_stitched_frame << "\n";

    std::cout << "when_show_on_the_screen: "
              << c.when_show_on_the_screen << "\n";

    std::cout << "==============================\n";
}
}

static uint8_t font8x8[256][8];
static bool inited = false;
static int thickness = 15;
static int half = thickness / 2;


void init_font8x8() {
    const uint8_t FONT_MINUS[8] = {0x00,0x00,0x00,0x7E,0x00,0x00,0x00,0x00};
    const uint8_t FONT_0[8]     = {0x3C,0x42,0x46,0x4A,0x52,0x62,0x3C,0x00};
    const uint8_t FONT_1[8]     = {0x08,0x18,0x08,0x08,0x08,0x08,0x3E,0x00};
    const uint8_t FONT_2[8]     = {0x3C,0x42,0x02,0x0C,0x30,0x40,0x7E,0x00};
    const uint8_t FONT_6[8]     = {0x3C,0x40,0x40,0x7C,0x42,0x42,0x3C,0x00};
    const uint8_t FONT_DEG[8]   = {0x18,0x24,0x24,0x18,0x00,0x00,0x00,0x00}; 
    // 清零
    memset(font8x8, 0, sizeof(font8x8));

    memcpy(font8x8['-'], FONT_MINUS, 8);
    memcpy(font8x8['0'], FONT_0, 8);
    memcpy(font8x8['1'], FONT_1, 8);
    memcpy(font8x8['2'], FONT_2, 8);
    memcpy(font8x8['6'], FONT_6, 8);
    memcpy(font8x8[0xB0], FONT_DEG, 8);
}

void draw_char_nv12_y(AVFrame* frame, int x, int y, char c, int Y_Y)
{
    uint8_t* Y = frame->data[0];
    int strideY  = frame->linesize[0];

    const uint8_t* bmp = font8x8[(uint8_t)c];

    for (int r = 0; r < 8; r++)
    {
        uint8_t row = bmp[r];

        for (int col = 0; col < 8; col++)
        {
            // 当前 bit 是否为 1（字符像素）
            if (row & (1 << (7 - col)))
            {
                // 放大 thickness × thickness
                for (int dy = 0; dy < thickness; dy++)
                {
                    for (int dx = 0; dx < thickness; dx++)
                    {
                        int xx = x + col * thickness + dx;
                        int yy = y + r * thickness + dy;

                        if (xx >= 0 && xx < frame->width &&
                            yy >= 0 && yy < frame->height)
                        {
                            Y[yy * strideY + xx] = Y_Y;
                        }
                    }
                }
            }
        }
    }
}

void draw_text_nv12(AVFrame* frame, int x, int y, const std::string& text, int fst, int Y)
{
    for (size_t i = 0; i < text.size(); i++) {
        draw_char_nv12_y(frame, x + i * (8 * thickness) - fst, y , text[i], Y);
    }
}

void draw_vertical_line_nv12(AVFrame *frame, int x, const std::string label, int fst, int Y_Y) {

    if(!inited) {
        init_font8x8();
        inited = true;
    }

    int W = frame->width;
    int H = frame->height;
    int offset = H * 0.1;

    // if (x < 0 || x >= W) return;

    // uint8_t* Y  = frame->data[0];
    // uint8_t* UV = frame->data[1];

    // int strideY  = frame->linesize[0];
    // int strideUV = frame->linesize[1];

    // for (int y = offset; y < H - offset; y++) {
    //     for (int dx = -half; dx <= half; dx++) {
    //         int xx = x + dx;
    //         if (xx >= 0 && xx < W)
    //             Y[y * strideY + xx] = Y_Y;
    //     }
    // }

    // --------------------
    // 写 label（放在上方）
    // --------------------
    int text_width = label.size() * 8;
    int text_x = x - text_width / 2;
    if (text_x < 0) text_x = 0;
    if (text_x + text_width >= W) text_x = W - text_width - 1;

    int text_y = offset - 200;   // 竖线上方
    
    draw_text_nv12(frame, text_x, text_y, label, fst, Y_Y);
}

