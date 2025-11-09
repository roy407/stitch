#include "config.h"
#include "log.hpp"

config::config() {
    loadFromFile("resource/hk5"); //需要修改，设置在一个文件夹下
}

bool config::loadFromFile(const std::string key) {
    std::string filename = key + ".json";
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        LOG_ERROR("Failed to open config file: {}" ,filename);
        return false;
    }

    json j;
    try {
        infile >> j;

        // 读取 global
        global.use_sub_input = j["global"]["use_sub_input"];
        global.software_status = j["global"]["software_status"];
        global.status = j["global"]["status"];
        global.save_rtsp_data_time = j["global"]["save_rtsp_data_time"];
        global.save_rtsp_data_path = j["global"]["save_rtsp_data_path"];

        // 读取 cameras
        for (const auto& cam : j["cameras"]) {
            CameraConfig c;
            c.name = cam["name"];
            std::string cam_status;
            if(global.use_sub_input) {
                cam_status = "sub";
            } else {
                cam_status = "main";
            }
            c.input_url = cam[cam_status.c_str()]["input_url"];
            c.width = cam[cam_status.c_str()]["width"];
            c.height = cam[cam_status.c_str()]["height"];
            c.output_url = cam["output_url"];

            // crop 是数组
            for (const auto& val : cam["crop"]) {
                c.crop.push_back(val);
            }

            c.rtsp = cam.value("rtsp", false); // 如果没有该字段则默认为 false

            // 解析 stitch
            if (cam.contains("stitch")) {
                const auto& s = cam["stitch"];
                c.stitch.enable = s.value("enable", false);
                c.stitch.mode = s.value("mode", "");
            }

            cameras.push_back(c);
        }

        // 读取 stitch
        stitch.output_url = j["stitch"]["output_url"];

        if(j["stitch"].contains("output_width")) {
            stitch.output_width = j["stitch"]["output_width"];
        } else {
            stitch.output_width = -1;
        }

        auto H_json = j["stitch"]["H_matrix_inv"];
        std::vector<std::array<double, 9>>& h_matrix_inv = stitch.h_matrix_inv;
        for (auto& [key, mat] : H_json.items()) {
            std::array<double, 9> arr{};
            int idx = 0;
            for (const auto& row : mat) {
                for (const auto& val : row) {
                    arr[idx++] = val.get<double>();
                }
            }
            h_matrix_inv.push_back(arr);
        }

        auto cam_polygons_json = j["stitch"]["cam_polygons"];
        std::vector<std::array<float, 8>>& cam_polygons = stitch.cam_polygons;
        for (auto& [key, points] : cam_polygons_json.items()) {
            std::array<float, 8> arr{};
            int idx = 0;

            for (const auto& p : points) {
                // 四个点，每个点两个坐标
                float x = static_cast<float>(std::round(p[0].get<double>()));
                float y = static_cast<float>(std::round(p[1].get<double>()));
                arr[idx++] = x;
                arr[idx++] = y;
            }
            cam_polygons.push_back(arr);
        }

    } catch (const std::exception& e) {
        LOG_WARN("parsing JSON failed, use default setting: {}",e.what());
        return false;
    }
    loadMappingTable(key, cameras[0].width * cameras.size(), cameras[0].height);
    return true;
}

bool config::loadMappingTable(const std::string key, uint64_t width, uint64_t height) {
    std::string filename = key + ".bin";
    std::ifstream infile(filename, std::ios::binary);
    if (!infile.is_open()) {
        LOG_ERROR("Failed to open config file: {}" ,filename);
        return false;
    }
    infile.seekg(0, std::ios::end);
    size_t file_bytes = (size_t)infile.tellg();
    infile.seekg(0, std::ios::beg);

    size_t expected_bytes = file_bytes;
    size_t expected_count = file_bytes / sizeof(uint16_t);
    std::vector<uint16_t> __buf(expected_count);
    infile.read(reinterpret_cast<char*>(__buf.data()), expected_bytes);

    size_t total = width * height;
    std::vector<MapEntry> buf(total);
    size_t src_idx = 0;
    for (uint64_t x = 0; x < width; ++x) {
        for (uint64_t y = 0; y < height; ++y) {
            size_t dst_idx = static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x);
            buf[dst_idx].cam_id = __buf[src_idx++]; // cam
            buf[dst_idx].map_x  = __buf[src_idx++]; // map_x
            buf[dst_idx].map_y  = __buf[src_idx++]; // map_y
            buf[dst_idx].pad    = 0;
        }
    }

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<ushort4>();
    cudaArray_t cuArray;
    cudaMallocArray(&cuArray, &channelDesc, width, height);
    cudaMemcpy2DToArray(
        cuArray,
        0, 0,
        buf.data(),
        width * sizeof(MapEntry),
        width * sizeof(MapEntry),
        height,
        cudaMemcpyHostToDevice);
    struct cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    // 配置纹理描述符
    struct cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;  // 不归一化坐标

    // 创建 texture 对象
    cudaCreateTextureObject(&d_mapping_table, &resDesc, &texDesc, nullptr);
    return true;
}

config& config::GetInstance() {  //是否线程安全？
    static config instance;
    return instance;
}

const GlobalConfig config::GetGlobalConfig() const {
    return global;
}

const std::vector<CameraConfig> config::GetCameraConfig() const {
    return cameras;
}

const GlobalStitchConfig config::GetGlobalStitchConfig() const {
    return stitch;
}

const cudaTextureObject_t config::GetMappingTable() const {
    return d_mapping_table;
}
