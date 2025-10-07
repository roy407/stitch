#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <atomic>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

struct StitchConfig {
    bool enable = false;
    std::string mode;
};

struct CameraConfig {
    std::string name;
    std::string input_url;
    int width;
    int height;
    std::string output_url;
    std::vector<float> crop;
    bool rtsp = false;
    StitchConfig stitch;
};

// 全局配置
struct GlobalConfig {
    bool use_sub_input;
    std::string software_status;
    std::string status;
    int save_rtsp_data_time;
    std::string save_rtsp_data_path;
};

// Stitch 配置
struct GlobalStitchConfig {
    std::string output_url;
    int output_width;
    std::vector<std::array<double, 9>> h_matrix;
};

// 配置管理类
class config {
private:
    GlobalConfig global;
    std::vector<CameraConfig> cameras;
    GlobalStitchConfig stitch;
    config();
    bool loadFromFile(const std::string& filename);
public:
    static config& GetInstance();
    const GlobalConfig GetGlobalConfig() const;
    const std::vector<CameraConfig> GetCameraConfig() const;
    const GlobalStitchConfig GetGlobalStitchConfig() const;
};
