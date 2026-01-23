#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <atomic>
#include <nlohmann/json.hpp>
#include <cuda_runtime.h>

using json = nlohmann::json;

struct CameraConfig {
    std::string name;
    int cam_id = 0;
    bool enable = true;
    std::string input_url;
    int width = 0;
    int height = 0;
    std::string output_url;
    bool enable_view = false;
    double scale_factor = 1.0;
    bool rtsp = false;
};

struct MappingTableConfig {
    std::string file_path;
    cudaTextureObject_t d_mapping_table;
    int output_width = 0;
};

struct StitchImplConfig {
    MappingTableConfig mapping_table;
    std::vector<std::array<double, 9>> h_matrix_inv;
    std::vector<std::array<float, 8>> cam_polygons;
};

struct StitchConfig {
    std::string stitch_mode;  
    StitchImplConfig stitch_impl;
    std::string output_url;
};

struct PipelineConfig {
    std::string name;
    int pipeline_id = 0;
    bool enable = true;
    bool use_substream = false;
    uint64_t default_width{0}; // 用于存储拼接图像的最大长度，json中不配置
    uint64_t default_height{0}; // 用于存储拼接图像的最大高度，json中不配置
    std::string main_stream; // 主码流相机数据文件
    std::string sub_stream; // 子码流相机数据文件
    std::vector<CameraConfig> cameras;
    StitchConfig stitch;
};

struct GlobalConfig {
    std::string loglevel;
    std::string type;
    std::string format;
    int record_duration = 0;
    std::string record_path;
    std::string decoder;
    std::string encoder;
    std::string model_path;
    std::string labels_path;
};

struct Config {
    GlobalConfig global;
    std::vector<PipelineConfig> pipelines;
};

struct MapEntry {
    uint16_t cam_id;
    uint16_t map_x;
    uint16_t map_y;
    uint16_t pad;  // 对齐为 8 字节
};

// 配置管理类
class config {
private:
    Config cfg;
    static std::string config_name;
    config();
    bool loadFromFile();
    void loadGlobalConfig(const json& j, GlobalConfig& cfg);
    void loadCamerasInfo(std::string file_path, PipelineConfig& pipe);
    void loadStitchConfig(const json& j, StitchConfig& stitch, uint64_t default_width, uint64_t default_height);
    void loadPipelineConfig(const json& j, PipelineConfig& pipe);
    bool loadMappingTable(cudaTextureObject_t& tex,
                            const std::string filename,
                            uint64_t width,
                            uint64_t height);
    void praseCameraConfig(const json& j, CameraConfig& cam);
public:
    static void SetConfigFileName(std::string cfg_name); // 一定要在初始化的时候就配置好
    static std::string GetConfigFileName();
    static config& GetInstance();
    const Config GetConfig() const;
    const std::string GetDetectionModelPath() const;
    const std::string GetDetectionLabelsPath() const;
    const GlobalConfig GetGlobalConfig() const;
    const PipelineConfig GetPipelineConfig(int pipeline_id) const;
    const std::vector<CameraConfig> GetCamerasConfig(int pipeline_id) const;
    const StitchConfig GetStitchConfig(int pipeline_id) const;
};

#define CFG_HANDLE config::GetInstance()
